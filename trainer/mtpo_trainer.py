import os

import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sequence

from typing import Any, Callable, Optional, Union, List, Dict, Tuple
from unittest.mock import patch

import pandas as pd
import numpy as np
from statistics import variance, mean
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import transformers
from accelerate.utils import broadcast_object_list, gather_object
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import pad
from trainer.agent import ReActAgent, dump_with_rich
from trainer.mtpo_config import MTPOConfig

if is_vllm_available():
    from vllm import LLM, SamplingParams

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


from torch.utils.tensorboard import SummaryWriter
import re
import random
from graphviz import Digraph 




class MTPOTrainer(Trainer):
    """
    MTPOTrainer that uses BFS-like expansions for ReAct steps.
    """
    _tag_names = ["trl", "mtpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        agent_cls: Optional[ReActAgent], 
        reward_funcs: Union[Callable, PreTrainedModel, str, List[Union[Callable, PreTrainedModel, str]]],
        args: MTPOConfig,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
    ):
        self.agent_cls = agent_cls
        # Prepare the MTPOConfig if needed
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = MTPOConfig(f"{model_name}-MTPO")

        self.depth = args.depth
        self.breadth = args.breadth

        self.writer = SummaryWriter(log_dir=args.output_dir)
        self._metrics = defaultdict(list)

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `MTPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `MTPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.use_vllm = args.use_vllm
        self.beta = args.beta
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )

                self.vllm_device = vllm_device
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=self.vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad checkpointing

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self.reward_funcs.append(self.llm_self_judge)

    def _set_signature_columns_if_needed(self):
        """
        Override so we only keep the columns we use in compute_loss (which is basically 'prompt', 'answer', etc.).
        """
        if self._signature_columns is None:
            self._signature_columns = ["question", "answer", "support_material_path"]

    def _prepare_inputs(self, inputs: dict) -> dict:
        # No default tokenization/cuda move is needed from Trainer, 
        # because we do custom logic in compute_loss
        return inputs
    
    # def compute_action_rewards(
    #     self,
    #     chains: List[List[dict]],
    #     reward_funcs: List[Callable[[str, Any], float]],
    #     ground_truth: Any,
    #     *,
    #     agg_leaf: Callable[[List[float]], float] | None = None,
    #     agg_internal: Callable[[List[float]], float] | None = None,
    # ) -> tuple[list[list[dict]], list[float]]:
    #     if agg_leaf     is None: agg_leaf     = max
    #     if agg_internal is None: agg_internal = lambda x: sum(x) / len(x)

    #     TAG_RE = re.compile(r'</?think>|</?answer>')
    #     CONTENT_RE = re.compile(
    #         r'^STEP-\d+:\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$',
    #         re.DOTALL | re.VERBOSE,
    #     )

    #     def total_repeated_chars(s: str) -> int:
    #         n = len(s)
    #         suffixes: List[Tuple[str, int]] = [(s[i:], i) for i in range(n)]
    #         suffixes.sort()
            
    #         intervals: List[Tuple[int, int]] = []
    #         for i in range(n - 1):
    #             a, ai = suffixes[i]
    #             b, bi = suffixes[i + 1]
    #             lcp = 0
    #             limit = min(len(a), len(b))
    #             while lcp < limit and a[lcp] == b[lcp]:
    #                 lcp += 1
    #             if lcp:
    #                 intervals.append((ai, ai + lcp))
    #                 intervals.append((bi, bi + lcp))

    #         if not intervals:
    #             return 0
    #         intervals.sort()
    #         merged = [list(intervals[0])]
    #         for start, end in intervals[1:]:
    #             if start > merged[-1][1]:
    #                 merged.append([start, end])
    #             else:
    #                 merged[-1][1] = max(merged[-1][1], end)
    #         return sum(end - start for start, end in merged)

    #     def format_reward_func(completion: str) -> float:
    #         if total_repeated_chars(completion) > 128:
    #             return 0.0
            
    #         step_match = re.match(r'^STEP-\d+:\r?\n', completion)
    #         if not step_match:
    #             return 0.0
    #         rest = completion[step_match.end():]
    #         if re.search(r'STEP-\d+:', rest):
    #             return 0.0

    #         think_match = re.match(r'<think>.*?</think>', rest, flags=re.DOTALL)
    #         if not think_match:
    #             return 0.0
    #         remaining = rest[think_match.end():].strip()
    #         if not remaining:
    #             return 0.5
    #         answer_pat   = r'^<answer>.*?</answer>$'
    #         toolcall_pat = r'^<tool_call>.*?</tool_call>$'
    #         if (re.fullmatch(answer_pat, remaining, flags=re.DOTALL) or
    #             re.fullmatch(toolcall_pat, remaining, flags=re.DOTALL)):
    #             return 0.5
    #         return 0.0

    #     # build tree
    #     node_children, node_parent_cnt, id2step = defaultdict(set), defaultdict(int), {}
    #     for chain in chains:
    #         for i, step in enumerate(chain):
    #             sid = id(step); id2step[sid] = step
    #             if i + 1 < len(chain):
    #                 cid = id(chain[i + 1])
    #                 node_children[sid].add(cid)
    #                 node_parent_cnt[cid] += 1
    #     root_ids = [sid for sid in id2step if node_parent_cnt[sid] == 0]

    #     # bottom-up
    #     from functools import lru_cache
    #     @lru_cache(maxsize=None)
    #     def dfs_raw(sid: int) -> float:
    #         step = id2step[sid]
    #         if not node_children[sid]:                                   # leaf
    #             r = agg_leaf([f(step.get("completions", ""), ground_truth) for f in reward_funcs])
    #         else:                                                        # internal
    #             r = agg_internal([dfs_raw(cid) for cid in node_children[sid]])
    #         step["reward"] = r
    #         return r
    #     for rid in root_ids:
    #         dfs_raw(rid)

    #     avg_acc = mean([id2step[rid]["reward"] for rid in root_ids])

    #     for step in id2step.values():
    #         if step.get("results"):
    #             if any("error" in str(sr) for sr in step["results"]):
    #                 step["reward"] *= 0.1
    #         step["reward"] += format_reward_func(step.get("completions", ""))

    #     # normalize
    #     depth_of = {}
    #     Q = deque([(rid, 0) for rid in root_ids])
    #     while Q:
    #         sid, d = Q.popleft()
    #         depth_of[sid] = d
    #         Q.extend((cid, d + 1) for cid in node_children[sid])

    #     depth_buckets = defaultdict(list)
    #     for sid, d in depth_of.items():
    #         depth_buckets[d].append(sid)

    #     for nodes in depth_buckets.values():
    #         vals = [id2step[sid]["reward"] for sid in nodes]
    #         mu = sum(vals) / len(vals)
    #         sigma2 = sum((v - mu) ** 2 for v in vals) / len(vals)
    #         sigma = sigma2 ** 0.5
    #         for sid in nodes:
    #             id2step[sid]["reward"] = 0.0 if sigma == 0 else (id2step[sid]["reward"] - mu) / sigma


    #     # render tree
    #     def render(sid: int, depth: int = 0) -> str:
    #         step  = id2step[sid]
    #         r     = step["reward"]
    #         txt   = f'{"  "*depth}- depth={depth} reward={r:.4f}\n'
    #         for cid in node_children[sid]:
    #             txt += render(cid, depth + 1)
    #         return txt

    #     with open(os.path.join(self.args.output_dir, "tmp_tree.txt"), "w", encoding="utf-8") as f:
    #         for rid in root_ids:
    #             f.write(render(rid))

    #     return avg_acc, chains

    def compute_action_rewards(
        self,
        chains: List[List[dict]],
        reward_funcs: List[Callable[[str, Any], float]],
        ground_truth: Any,
        *,
        agg_leaf: Callable[[List[float]], float] | None = None,
        agg_internal: Callable[[List[float]], float] | None = None,
    ) -> tuple[float, List[List[dict]]]:

        if agg_leaf is None:     agg_leaf = max
        if agg_internal is None: agg_internal = lambda xs: sum(xs) / len(xs)
        GRAPH_FMT, IMG_NAME = "svg", "tmp_tree.svg"

        # ---------- helpers ----------
        def total_repeat(s: str) -> int:
            n, suf = len(s), sorted((s[i:], i) for i in range(len(s)))
            iv = []
            for i in range(n - 1):
                a, ai = suf[i]
                b, bi = suf[i + 1]
                lcp = 0
                while lcp < min(len(a), len(b)) and a[lcp] == b[lcp]:
                    lcp += 1
                if lcp:
                    iv += [(ai, ai + lcp), (bi, bi + lcp)]
            if not iv: return 0
            iv.sort()
            m = [list(iv[0])]
            for s0, e0 in iv[1:]:
                if s0 > m[-1][1]:
                    m.append([s0, e0])
                else:
                    m[-1][1] = max(m[-1][1], e0)
            return sum(e - s for s, e in m)

        def fmt_bonus(c: str) -> float:
            if total_repeat(c) > 128:                    return 0.0
            if not re.match(r'^STEP-\d+:\r?\n', c):      return 0.0
            rest = re.sub(r'^STEP-\d+:\r?\n', '', c, 1)
            if re.search(r'STEP-\d+:', rest):            return 0.0
            think = re.match(r'<think>.*?</think>', rest, re.S)
            if not think:                                return 0.0
            remain = rest[think.end():].strip()
            if not remain:                               return 0.5
            ok = r'^<answer>.*?</answer>$|^<tool_call>.*?</tool_call>$'
            return 0.5 if re.fullmatch(ok, remain, re.S) else 0.0

        def green(val: float) -> str:
            val = max(0, min(1, val))
            lg, dg = (0xE0, 0xFF, 0xE0), (0x00, 0x64, 0x00)
            rgb = [int(l + val * (d - l)) for l, d in zip(lg, dg)]
            return f'#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}'

        # ---------- build DAG ----------
        ch, par, id2 = defaultdict(set), defaultdict(int), {}
        for chain in chains:
            for i, st in enumerate(chain):
                sid = id(st); id2[sid] = st
                if i + 1 < len(chain):
                    cid = id(chain[i + 1])
                    ch[sid].add(cid); par[cid] += 1
        roots = [sid for sid in id2 if par[sid] == 0]

        # ---------- raw rewards ----------
        from functools import lru_cache
        @lru_cache(None)
        def dfs(sid: int) -> float:
            st = id2[sid]
            if not ch[sid]:
                r = agg_leaf([f(st.get("completions", ""), ground_truth)
                            for f in reward_funcs])
            else:
                r = agg_internal([dfs(c) for c in ch[sid]])
            st["reward"] = r
            return r
        for r in roots: dfs(r)

        avg_acc = mean(id2[r]["reward"] for r in roots)

        # ---------- min–max by prompt ----------
        bucket = defaultdict(list)
        for sid, st in id2.items():
            bucket[st.get("prompt", "<none>")].append(sid)
        for sids in bucket.values():
            vs = [id2[s]["reward"] for s in sids]
            vmin, vmax = min(vs), max(vs); rng = vmax - vmin
            for s in sids:
                id2[s]["reward"] = id2[s]["reward"] if rng == 0 \
                                else (id2[s]["reward"] - vmin) / rng

        # ---------- add super root ----------
        super_root = {"reward": avg_acc}
        super_id = id(super_root)
        id2[super_id] = super_root
        ch[super_id] = set(roots)
        roots = [super_id]

        # ---------- early render BEFORE penalties ----------
        def render_graph(path_prefix: str):
            dot = Digraph(format=GRAPH_FMT); dot.attr(rankdir="TB")
            dot.attr('node', shape='box', style='filled', fontsize="10")
            for sid, st in id2.items():
                v = st["reward"]; dot.node(str(sid), f"{v:.2f}", fillcolor=green(v))
            for p, cs in ch.items():
                for c in cs: dot.edge(str(p), str(c))
            dot.render(path_prefix, directory=self.args.output_dir, cleanup=True)
            
        render_graph(IMG_NAME.rsplit(".", 1)[0])  # writes tmp_tree.svg

        # ---------- penalties + bonus ----------
        for st in id2.values():
            if st.get("results") and any("error" in str(r) for r in st["results"]):
                st["reward"] *= 0.1
            st["reward"] += fmt_bonus(st.get("completions", ""))

        return avg_acc, chains
    
    def llm_self_judge(self, model_output, ground_truth):
        if str(ground_truth) not in model_output:
            return 0.0

        matches = re.findall(r'<answer>(.*?)</answer>', model_output)
        if matches:
            model_output = matches[-1]
        else:
            return 0.0

        prompt = f"""\
Evaluate the model’s answer against the human-annotated ground truth.

## Instructions
1. Return a correctness score **either 0 or 1** (1 represents model_output == ground_truth).  
3. Wrap **only** the final score in `<answer>…</answer>`.  

## Query
{self.question.split('👆')[0]}

## Model Output
{model_output}

## Ground Truth
{ground_truth}"""
        msg = [{"role":"user", "content": prompt}]
        prompt = self.processing_class.apply_chat_template(
            conversation=msg, 
            tokenize=False, 
            add_generation_prompt=True
        )
        generation_result = self.llm.generate(
            prompts=[prompt], 
            sampling_params=self.sampling_params, 
            use_tqdm=False
        )
        output_obj = generation_result[0].outputs[0]
        token_ids = output_obj.token_ids
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)

        result = self.processing_class.decode(token_ids, skip_special_tokens=True)

        with open(os.path.join(self.args.output_dir,"tmp_llm_self_judge.txt"), "w") as f:
            f.write("="*20 + "\n"); f.write(prompt)
            f.write("="*20 + "\n"); f.write(result)
        
        matches = re.findall(r'<answer>(.*?)</answer>', result)
        if matches:
            try:
                score = float(matches[-1])
                if score == 1.0:
                    return 0.5
                else:
                    return 0.0
            except Exception as e:
                return 0.0
        else:
            return 0.0

    def _slow_eval_on_step(self, num_samples: int = 100):
        if self.eval_dataset is None or not self.accelerator.is_main_process:
            return

        # turn it into a real Python sequence if needed
        if isinstance(self.eval_dataset, Sequence):
            population = self.eval_dataset
        else:
            # * datasets.Dataset / torch Dataset: build a lightweight list of indices
            population = list(range(len(self.eval_dataset)))

        sample_size = min(num_samples, len(population))
        sample_idxs = random.sample(population, sample_size)

        # If we sampled indices, fetch the rows
        batch = (
            [self.eval_dataset[i] for i in sample_idxs]
            if population is not self.eval_dataset
            else [self.eval_dataset[i] for i in sample_idxs]  # same path, explicit
        )

        rewards = list()
        for inp in batch:
            self.question, self.ground_truth, self.support_material_path  = inp["question"], inp["ground_truth"], inp["support_material_path"]

            agent = self.agent_cls(
                tokenizer =       self.processing_class,
                depth =           self.depth,
                breadth =         1,
                llm =             self.llm,
                device =          self.vllm_device, 
                sampling_params = self.sampling_params,
                output_dir =      self.args.output_dir
            )
            last_stp = agent.react_recursive(
                    support_material_path = self.support_material_path,
                    question =         self.question, 
                    ground_truth =     self.ground_truth
                )[0][-1]
            rewards.append(
                max([rwf(last_stp.get("completions", ""), self.ground_truth) for rwf in self.reward_funcs])
            )
        if rewards:
            avg_r = float(np.mean(rewards))
            self._metrics["eval_acc"].append(avg_r)
            self.writer.add_scalar("Eval/EvalAcc", avg_r, self.state.global_step)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("MTPOTrainer does not support `return_outputs=True`.")

        device = self.accelerator.device
        if self.state.global_step != self._last_loaded_step:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                state_dict = unwrapped.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            self._last_loaded_step = self.state.global_step

        # BFS expansions per sample
        if self.accelerator.is_main_process:
            batched_scored_chains = []
            avg_acc_list = []
            for inp in inputs:
                self.question, self.ground_truth, self.support_material_path  = inp["question"], inp["ground_truth"], inp["support_material_path"]

                react_agent = self.agent_cls(
                    tokenizer =       self.processing_class,
                    depth =           self.depth,
                    breadth =         self.breadth,
                    llm =             self.llm,
                    device =          self.vllm_device,
                    sampling_params = self.sampling_params,
                    output_dir =      self.args.output_dir,
                )
                expansions = react_agent.react_recursive(
                    support_material_path = self.support_material_path, 
                    question =              self.question, 
                    ground_truth =          self.ground_truth
                )

                avg_acc, scored_chains = self.compute_action_rewards(
                    chains=expansions,
                    ground_truth=self.ground_truth,
                    reward_funcs=self.reward_funcs,
                )
                avg_acc_list.append(avg_acc)
                batched_scored_chains.extend(
                    scored_chains
                )

            # flatten, de-duplicate by completion text
            seen, flattened_steps = set(), []
            for chain in batched_scored_chains:
                for step in chain:
                    key = step.get("completions", "")
                    if key not in seen:
                        flattened_steps.append(step)
                        seen.add(key)

            # pick depth with max reward variance, then quantile-sample
            buckets_raw = defaultdict(list)
            for s in flattened_steps:
                buckets_raw[s["current_depth"]].append(s)

            batch_avg_acc = mean(avg_acc_list)
            
            self._metrics["avgAcc"].append(batch_avg_acc)
            self.writer.add_scalar("avgAcc", batch_avg_acc, self.state.global_step)

            buckets = {
                d: v for d, v in buckets_raw.items() if len(v)>1 and variance([s["reward"] for s in v]) > 0
            }

            if not buckets:
                print(
                    f"[MTPOTrainer] step={self.state.global_step}  "
                    "all reward variances == 0 → skip update, loss = 0"
                )
                for d, v in buckets_raw.items():
                    print(f"DEPTH: {d}; \n REWARD_LIST: {[s['reward'] for s in v]}") 
                flattened_steps = list()
            else:
                candidates = [step for layer in buckets.values() for step in layer if len(step["completion_ids"]) < self.max_completion_length]
                print(f"We have {len(buckets)} layers with non-zero variance, total candidates: {len(candidates)}")

                if len(candidates) > self.num_generations:
                    # flattened_steps = random.sample(candidates, self.num_generations)
                    idx = np.linspace(0, len(candidates) - 1, self.num_generations, dtype=int)
                    flattened_steps = [candidates[i] for i in idx]
                else:
                    flattened_steps = candidates

                print(f"Sampled {len(flattened_steps)} steps for optimization")

                
            for fstp in flattened_steps:
                dump_with_rich(fstp, os.path.join(self.args.output_dir, "tmp_train.txt"))
                fstp.pop("prompt", None)
                fstp.pop("completions", None)
                fstp.pop("results", None)
                fstp.pop("ground_truth", None)

            if flattened_steps:
                print(flattened_steps[0].keys())

        # Broadcast flattened_steps to every rank
        wrapper = [flattened_steps] if self.accelerator.is_main_process else [None]
        wrapper = broadcast_object_list(wrapper, from_process=0); flattened_steps = wrapper[0]

        # abort early if nothing to train on
        if not flattened_steps:
            self._metrics["loss"].append(0.0)
            return torch.tensor(0.0, device=device, requires_grad=True)
        group_size = len(flattened_steps)

        completion_ids = [torch.tensor(stp["completion_ids"], device=device) for stp in flattened_steps]
        # Pad the completions, and concatenate them with the prompts
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_inputs_repeated = torch.repeat_interleave(flattened_steps[0]["prompt_ids"], group_size, dim=0).to(device)
        prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)

        prompt_length = flattened_steps[0]["prompt_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # 5. Batched log-prob collection
        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, num_logits_to_keep):
            # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        num_logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, num_logits_to_keep)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Pad/mask & loss
        rewards = torch.tensor([stp["reward"] for stp in flattened_steps], requires_grad=True).to(device)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, group_size).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, group_size).std(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, group_size).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, group_size).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(group_size, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(group_size, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        comp_len = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(comp_len)
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(mean_grouped_rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(1) / completion_mask.sum(1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["loss"].append(loss.item())

        if hasattr(self, "writer"):
            step_id = self.state.global_step
            self.writer.add_scalar("Loss/PolicyLoss", loss.item(), step_id)
            self.writer.add_scalar("Metrics/TrainReward", rewards.mean().item(), step_id)
            self.writer.add_scalar("Metrics/KL", mean_kl.item(), step_id)
        # self._slow_eval_on_step(num_samples=50)

        return loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        Overridden to incorporate the metrics we collected. 
        """
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()} if self._metrics else {}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()
