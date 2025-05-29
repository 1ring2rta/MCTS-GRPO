import json
import re
import os
import copy
import traceback
import torch
import pandas as pd
import gc
import abc

from typing import ClassVar, Optional, List, Dict, Any
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
        # self.tools_description = [
        #     {
        #     "type": "function",
        #     "function": {
        #             "name": "exe_lambda",
        #             "description": "Lambda Expression Executor.",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "packages": {
        #                             "type": "string",
        #                             "description": "Import statements needed prior to execution, e.g. 'import pandas as pd; import numpy as np'."
        #                         },
        #                     "lambda_expression": {
        #                             "type": "string",
        #                             "description": "A valid Python lambda expression with variables all from globals(), e.g. `lambda: a+b`"
        #                         },
        #                     },
        #                 "required": [
        #                     "lambda_expression"
        #                 ]
        #             }
        #         }
        #     }
        # ]

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

            # if len(token_ids) == self.sampling_params.max_tokens:
            #     raise RuntimeError("max tokens exceeded，cut by LLM.generate.")

            if not isinstance(token_ids, list):
                token_ids = list(token_ids)
            
            completions = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            if len(token_ids) >= self.sampling_params.max_tokens:
                completions += "... (max token exceeded, try to be brief)"
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
            support_material, support_material_str = None, None
        return support_material, support_material_str

    def react_recursive(
        self,
        question: str,
        support_material_path: Optional[list] = None,
        ground_truth: Optional[str] = None, 
        assistant_and_tool_msg: Optional[list] = None,
        current_chain: Optional[list] = None,
        current_depth: int = 0,
    ) -> list:
        support_material, support_material_str = self.support_material_read_func(support_material_path)

        assistant_and_tool_msg = copy.deepcopy(assistant_and_tool_msg) if assistant_and_tool_msg else []
        current_chain = current_chain if current_chain else []

        current_results = {}
        for step_resp in current_chain:
            for i, res in enumerate(step_resp["results"]):
                current_results[f"result_{len(current_results)}"] = res

        support_material_str = f"# Support Material:\n{support_material_str}" if support_material_str else ""
#         prompt = f"""\
# - Gather sufficient information or perform necessary verifications by invoking relevant tools.
# - Provide comprehensive reasoning by clearly outlining your chain of thought.
# - Conclude by presenting a definitive answer to exit the loop.


# {support_material_str}


# # Notice:
# 1. Each response **MUST contain exactly one** `<think>...</think>` block.  
#    • If tool usage is needed, it **must be immediately followed** by one `<tool_call>...</tool_call>` block.  
#    • If the final answer is ready, it **must be immediately followed** by one `<answer>...</answer>` block.  
#    • A single response **MUST NOT contain both** `<tool_call>` and `<answer>`.  
#    • No additional visible content is allowed outside these tags (only whitespace is permitted).
# 2. Inside `<think>...</think>`:  
#    • Clearly explain your reasoning and justify your next step.  
#    • DO NOT include any nested `<think>` or `<answer>` tags.
# 3. Inside `<tool_call>...</tool_call>`:  
#    • Include only when necessary, and always place it directly after `<think>`.
# 4. Inside `<answer>...</answer>`:  
#    • Provide the final answer to the user and conclude the reasoning process.
# 5. Tool call format example (must be preceded by a `<think>` block):
# <think>
# I need to look up the value of the 'third' field for season 2008. I will call exe_lambda.
# </think>
# <tool_call>
# {{"name": "exe_lambda", "arguments": {{"packages": "import pandas as pd", "lambda_expression": "lambda: df0[df0['season'] == 2008]['third'].values[0]"}}}}
# </tool_call>
# 6. Please provide your final answer in {self.depth} steps. You are currently on step {current_depth+1} of {self.depth}.

# # User Question: {question}"""
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
                    res_key = f"result_{len(current_results)}"
                    current_results[res_key] = e
                    resp["results"].append(e)
                    local_msgs.append({
                        "role": "tool",
                        "name": "~",
                        "content": f"Var: {res_key}; Type: {type(e)}\n{e}"
                    })
                local_msgs.append(assistant_msg)

                tool_calls = assistant_msg.get("tool_calls", [])
                if len(tool_calls) > 1:
                    raise RuntimeError(f"Expected at most 1 tool calls, but got {len(tool_calls)}")

                for call in tool_calls:
                    tool_name = call["function"]["name"]
                    tool_args = call["function"]["arguments"] or {}

                    if tool_name not in self.TOOLS:
                        raise ValueError(f"Unknown tool: {tool_name}")
                    
                    context = {**current_results, **support_material}
                    try:
                        result = self.TOOLS[tool_name](**tool_args, context=context)
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
                    "name": "~",
                    "content": f"Var: {res_key}; Type: {type(e)}\n{e}"
                })

            local_chain.append(resp)
            if current_depth+1 < self.depth:
                sub_chains = self.react_recursive(
                    question=question,
                    support_material_path=support_material_path,
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
