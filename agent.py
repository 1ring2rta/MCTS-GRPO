import json
import re
import os
import copy
import traceback
import torch
import pandas as pd
import gc

from typing import Optional
from vllm import LLM, SamplingParams
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    GenerationConfig,
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
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        func = json.loads(m.group(1))
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


def exe_lambda(lambda_expression: str, packages=None, inputs=None, context:dict=dict()):
    if packages is not None:
        exec(packages)
        context.update(locals())

    resolved_inputs = {}
    if inputs:
        for input_name, var_name in inputs.items():
            if var_name not in context:
                raise KeyError(f"Variable '{var_name}' not found in context.")
            resolved_inputs[input_name] = context[var_name]

    result = eval(lambda_expression, context)(**resolved_inputs)
    return result


class ReActAgent:
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
        self.tools = [
            {
            "type": "function",
            "function": {
                    "name": "exe_lambda",
                    "description": "Lambda Expression Executor.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "packages": {
                                    "type": "string",
                                    "description": "Import statements needed prior to execution, e.g. 'import pandas as pd; import numpy as np'."
                                },
                            "lambda_expression": {
                                    "type": "string",
                                    "description": "A valid Python lambda expression with variables all from globals(), e.g. `lambda: a+b`"
                                },
                            },
                        "required": [
                            "lambda_expression"
                        ]
                    }
                }
            }
        ]

    def _generate(self, messages: list[dict], ground_truth) -> dict:
        with torch.inference_mode():
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages, 
                tools=self.tools, 
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

            # if len(token_ids) == self.sampling_params.max_tokens:
            #     raise RuntimeError("max tokens exceeded，cut by LLM.generate.")

            if not isinstance(token_ids, list):
                token_ids = list(token_ids)

            completions = self.tokenizer.decode(token_ids, skip_special_tokens=True)
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

    def react_recursive(
        self,
        query: str,
        given: Optional[dict] = None,
        ground_truth = "", 
        assistant_and_tool_msg: Optional[list] = None,
        current_chain: Optional[list] = None,
        current_depth: int = 0,
    ) -> list:
        given = {} if given is None else given
        assistant_and_tool_msg = copy.deepcopy(assistant_and_tool_msg) if assistant_and_tool_msg else []
        current_chain = current_chain if current_chain else []

        current_results = {}
        for step_resp in current_chain:
            for i, res in enumerate(step_resp["results"]):
                current_results[f"result_{len(current_results)}"] = res

        given_str = "\n".join(
            f"Var: {k}; Type: {type(v)}\n{v}" + f"\n{v.dtypes}" if isinstance(v, pd.DataFrame) else f"Var: {k}; Type: {type(v)}\n{v}" for k, v in given.items()
        )
        prompt = f"""\
- Gather sufficient information or perform necessary verifications by invoking relevant tools.
- Provide comprehensive reasoning by clearly outlining your chain of thought.
- Conclude by presenting a definitive answer to exit the loop.

# Given:
{given_str}

# Notice:
1. Think before you call tools or answer, use <think>...</think> to perform your chain of thoughts (CoT).
2. Once you are confident in the results, use <answer>...</answer> to provide your final answer and end the loop.
3. Here is an tool_call example:
<tool_call>
{{"name": "exe_lambda", "arguments": {{"packages": "import pandas as pd", "lambda_expression": "lambda: df0[df0['season'] == 2008]['third'].values[0]"}}}}                                    │
</tool_call>
4. Please provide your final answer in {self.depth} steps. You are currently on step {current_depth+1} of {self.depth}.


# User Query: {query}"""

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
                    res_key = f"result_{len(current_results)}"
                    current_results[res_key] = e
                    resp["results"].append(e)
                    local_msgs.append({
                        "role": "tool",
                        "name": "--",
                        "content": f"Var: {res_key}; Type: {type(e)}\n{e}"
                    })
                local_msgs.append(assistant_msg)

                tool_calls = assistant_msg.get("tool_calls", [])
                if len(tool_calls) > 2:
                    raise RuntimeError(f"Expected at most 2 tool calls, but got {len(tool_calls)}")

                for call in tool_calls:
                    tool_name = call["function"]["name"]
                    tool_args = call["function"]["arguments"] or {}

                    context = {**current_results, **given}
                    try:
                        result = eval(tool_name)(**tool_args, context=context)
                    except Exception as e:
                        result = e

                    resp["results"].append(result)
                    res_key = f"result_{len(current_results)}"
                    current_results[res_key] = result

                    local_msgs.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Var: {res_key}; Type: {type(result)}\n{result}"
                    })

            except Exception as e:
                res_key = f"result_{len(current_results)}"
                current_results[res_key] = e
                resp["results"].append(e)
                local_msgs.append({
                    "role": "tool",
                    "name": "--",
                    "content": f"Var: {res_key}; Type: {type(e)}\n{e}"
                })

            local_chain.append(resp)
            if current_depth + 1 < self.depth:
                sub_chains = self.react_recursive(
                    query=query,
                    given=given,
                    ground_truth=ground_truth, 
                    assistant_and_tool_msg=local_msgs,
                    current_chain=local_chain,
                    current_depth=current_depth + 1,
                )
                all_chains.extend(sub_chains)
            else:
                all_chains.append(local_chain)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return all_chains