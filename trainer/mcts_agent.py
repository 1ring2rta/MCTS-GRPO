import json
import re
import os
import copy

import torch
import pandas as pd

import gc
import abc
import traceback

from typing import ClassVar, Optional, List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import (
    PreTrainedTokenizerBase,
)

from rich.console import Console
from rich.panel import Panel
from rich.markup import escape

from io import StringIO
from pathlib import Path



def dump_with_rich(step: dict, logfile: str):
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, record=False)

    console.print(Panel(escape(str(step["reward"])), title="REWARD"))
    console.print(Panel(escape(str(step["prompt"])), title="PROMPT"))
    console.print(Panel(escape(str(step["completions"])), title="COMPLETION"))
    console.print(Panel(escape(str(step["ground_truth"])), title="GROUND TRUTH"))

    text = buf.getvalue()
    Path(logfile).write_text(text, encoding="utf-8")
    return logfile


def parse_tool_calls(content:str):
    tool_calls = []
    offset = 0
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for i, m in enumerate(pattern.finditer(content)):
        if i == 0:
            offset = m.start()
        func = json.loads(m.group(1).strip().replace("\n", "\\n"))
        tool_calls.append({"type": "function", "function": func})
        if isinstance(func["arguments"], str):
            func["arguments"] = json.loads(func["arguments"])
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else: 
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}


class ReActAgent(abc.ABC):

    ACTION_PROMPT_TEMPLATE: ClassVar[str]
    TOOLS:                  ClassVar[Dict[str, callable]]
    TOOLS_DESCRIPTION:      ClassVar[List[Dict[str, Any]]]

    def __init__(
        self, 
        tokenizer: Optional[PreTrainedTokenizerBase], 
        depth: int, 
        breadth: int, 
        output_dir: str, 
        llm: Optional[LLM] = None, 
        device = torch.device("cuda"), 
        sampling_params: Optional[SamplingParams] = None, 
    ):
        self.tokenizer = tokenizer
        self.depth = depth
        self.breadth = breadth
        self.output_dir = output_dir
        self.llm = llm
        self.device = device
        self.sampling_params = sampling_params

    def _generate(self, messages: list[dict], ground_truth) -> dict:
        with torch.inference_mode():
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages, 
                tools=self.TOOLS_DESCRIPTION, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompt_enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            prompt_ids = prompt_enc["input_ids"]

            generation_result = self.llm.generate(
                prompts=[prompt],
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            output_obj = generation_result[0].outputs[0]
            token_ids = output_obj.token_ids

            if not isinstance(token_ids, list):
                token_ids = list(token_ids)
            
            completions = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            if len(token_ids) >= self.sampling_params.max_tokens:
                completions += "... (max tokens exceeded)"
            result = {
                "prompt": prompt,
                "completions": completions,
                "prompt_ids": prompt_ids, 
                "completion_ids": token_ids,
                "ground_truth": ground_truth, 
                "reward": None
            }
            dump_with_rich(result, os.path.join(self.output_dir, "tmp.txt"))

            return result

    def support_material_read_func(self, table_paths):
        if table_paths:
            support_material = dict()
            for i in range(len(table_paths)):
                try:
                    support_material[f"df{i}"] = pd.read_csv(table_paths[i])
                except Exception as e:
                    with open(table_paths[i]) as f: support_material[f"tb{i}"] = f.read()

            support_material_str = "\n".join(
                f"Var: {k}; Type: {type(v)}\n{v}" + f"\n{v.dtypes}" if isinstance(v, pd.DataFrame) else f"Var: {k}; Type: {type(v)}\n{v}" for k, v in support_material.items()
            )
        else:
            support_material, support_material_str = dict(), str()
        return support_material, support_material_str

    def react_recursive(
        self,
        question: str,
        support_material_path: Optional[list] = None,
        ground_truth: Optional[str] = None, 
        assistant_and_tool_msg: Optional[list] = None,
        current_chain: Optional[list] = None,
        current_depth: int = 0,
        previous_variables: dict = dict()
    ) -> list:
        support_material, support_material_str = self.support_material_read_func(support_material_path)

        assistant_and_tool_msg = copy.deepcopy(assistant_and_tool_msg) if assistant_and_tool_msg else []
        current_chain = current_chain if current_chain else []

        support_material_str = f"# Support Material:\n{support_material_str}" if support_material_str else ""
        prompt = self.ACTION_PROMPT_TEMPLATE.format(
            support_material_str=support_material_str,
            question=question,
            max_steps=self.depth, 
            current_step=current_depth+1
        )

        user_msg = [{"role": "user", "content": prompt}]
        responses = [self._generate(messages=user_msg+assistant_and_tool_msg, ground_truth=ground_truth) for _ in range(self.breadth)]
        all_chains = []

        for resp in responses:
            resp["current_depth"] = current_depth

            local_msgs = copy.deepcopy(assistant_and_tool_msg)
            local_chain = current_chain.copy()
            resp["results"] = []

            try:
                if "<answer>" in resp["completions"] or "<tool_call>" not in resp["completions"]:
                    local_chain.append(resp)
                    all_chains.append(local_chain)
                    continue
                
                try:
                    assistant_msg = parse_tool_calls(resp["completions"])
                except Exception as e:
                    assistant_msg = {
                        "role": "assistant",
                        "content": resp["completions"]
                    }
                    resp["results"].append({"parse_error": e})
                    local_msgs.append({
                        "role": "tool",
                        "name": "none",
                        "content": f"Var: e; Type: {type(e)}\n{e}"
                    })
                local_msgs.append(assistant_msg)

                tool_calls = assistant_msg.get("tool_calls", [])
                # if len(tool_calls) > 1:
                #     raise RuntimeError(f"Expected at most 1 tool calls, but got {len(tool_calls)}")

                
                for call in tool_calls:
                    tool_name = call["function"]["name"]
                    tool_args = call["function"]["arguments"] or {}

                    if tool_name not in self.TOOLS:
                        raise ValueError(f"Unknown tool: {tool_name}")
                    
                    context = {**previous_variables, **support_material}
                    try:
                        output_str, new_context = self.TOOLS[tool_name](**tool_args, context=context)
                    except Exception as e:
                        output_str, new_context = f"Var: tool_execute_error; Type: {type(e)}\n{traceback.format_exc()}", context

                    stp_variables = {k: v for k,v in new_context.items() if k not in context}
                    resp["results"].append({f"tool_call[{tool_calls.index(call)}]": stp_variables})
                    local_msgs.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": output_str
                    })
                    previous_variables = {k: v for k,v in new_context.items() if k not in support_material}

            except Exception as e:
                resp["results"].append({"e": e})
                local_msgs.append({
                    "role": "tool",
                    "name": "none",
                    "content": f"Var: error; Type: {type(e)}\n{traceback.format_exc()}"
                })

            local_chain.append(resp)
            if current_depth+1 < self.depth:
                sub_chains = self.react_recursive(
                    question               = question,
                    support_material_path  = support_material_path,
                    ground_truth           = ground_truth, 
                    assistant_and_tool_msg = local_msgs,
                    current_chain          = local_chain,
                    current_depth          = current_depth + 1,
                    previous_variables     = previous_variables
                )
                all_chains.extend(sub_chains)
            else:
                all_chains.append(local_chain)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return all_chains
