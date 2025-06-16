from typing import Dict, List
from types import ModuleType
from tools.local_python_executor import LocalPythonExecutor
from tools.tool_base import Tool

import re
import json
import pandas as pd
import numpy as np
import textwrap
import reprlib

import signal
from logging import getLogger
from contextlib import contextmanager


pa = r'(\b[\w\d\(\)\+\-\*/\s]+\b)\s*\*\*\s*(\d{6,}|(\(*\s*\d+\s*\**\s*\**\s*\d+\s*\)*))'

DEFAULT_VARS = {
}
logger = getLogger(__file__)


class TimeoutException(Exception):
    """Raised when code execution exceeds the allowed time."""
    pass


def smart_repr(obj, *, list_items=10, str_chars=120):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        info = f"<{type(obj).__name__} shape={obj.shape}>"
        # head & tail as a mini-preview
        preview = (
            obj.head(3).to_string(max_rows=3, max_cols=10, show_dimensions=False)
            + ("\n...\n" + obj.tail(3).to_string(max_rows=3, max_cols=10, show_dimensions=False)
               if len(obj) > 6 else "")
        )
        return f"{info}\n{preview}"

    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>\n{np.array2string(obj, threshold=20)}"

    if isinstance(obj, str) and len(obj) > str_chars:
        return textwrap.shorten(obj, width=str_chars, placeholder="…")

    if isinstance(obj, (list, tuple, set)):
        # slice & keep symmetry: first N//2, last N//2
        seq = list(obj)
        if len(seq) > list_items:
            half = list_items // 2
            shown = seq[:half] + ["…"] + seq[-half:]
        else:
            shown = seq
        return f"{type(obj).__name__}({shown})"

    if isinstance(obj, dict):
        if len(obj) > list_items:
            items = list(obj.items())
            half = list_items // 2
            shown = items[:half] + [("…", "…")] + items[-half:]
        else:
            shown = obj.items()
        return f"dict({shown})"

    r = reprlib.Repr()
    r.maxstring = str_chars
    r.maxother  = str_chars
    return r.repr(obj)

def format_variables(variables: dict) -> str:
    """
    Build the nicely truncated dump string.
    """
    pieces = []
    for k, v in list(variables.items())[-10:]:
        if not isinstance(v, ModuleType):
            pieces.append(f"Var: {k}; Type: {type(v).__name__}\n{smart_repr(v)}")
    return "\n".join(pieces)


@contextmanager
def time_limit(seconds: int):
    """
    Context manager to limit execution time of a code block.
    Works on POSIX systems that support SIGALRM.
    """
    def _handle_timeout(signum, frame):
        raise TimeoutException(f"Execution exceeded {seconds}s time limit")

    original_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original_handler)


class IPythonInterpreter(Tool):
    """
    Tool for executing Python code similar to a Jupyter notebook
    """
    
    def __init__(
            self, 
            local_dict, 
            default_timeout: int = 10
        ):
        
        self.default_timeout = default_timeout
        
        name = "ipython_interpreter"
        description = "Execute Python code and return the results, use print() to get detailed infomathon of varibales."
        parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
            },
            "required": ["code"]
        }
        
        super().__init__(name, description, parameters)
        
        # Initialize the execution environment
        self.globals_dict = {}
        self.locals_dict = local_dict if local_dict else DEFAULT_VARS

        self._is_finish = False
        
        # Import commonly used libraries
        # ...
        
        # Initialize python code executor
        self.executor = LocalPythonExecutor(
            additional_authorized_imports=["*"],
        )
        self.executor.send_tools({})
    
    def execute(self, code: str, timeout: int | None = None) -> dict:
        if self.locals_dict:
            self.executor.send_variables(self.locals_dict)

        timeout = self.default_timeout if timeout is None else timeout
        try:
            with time_limit(timeout):
                self.executor(code)
            return {"status": "success", "results": self.executor.state}

        except TimeoutException as e:
            self._is_finish = False
            return {"status": "failed", "results": str(e)}
        except Exception as e:
            self._is_finish = False
            return {"status": "failed", "results": str(e)}


    def batch_execute(self, args_list: List[Dict[str, str]]) -> List[str]:
        results = []
        for args in args_list:
            code:str = args.get('code', '')
            if isinstance(code,str):
                match = re.findall(pa,code)
                if match:
                    logger.error(f"Can not get code: {args}")
                    results.append({"status": 'failed', "result": f'Do not calculate the {match[0]} directily, much use the packages or cause OOM'})
                    continue
            if not code:
                logger.error(f"Can not get code: {args}")

            results.append(self.execute(code))
        return results

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for code execution
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            result_obj = json.loads(result)
            if result_obj.get("status") == "success":
                return 0.1  # Reward successful execution
            else:
                return -0.1  # Small penalty for execution errors
        except:    # noqa: E722
            return -0.1  # Penalty for invalid result format

    def reset_environment(self):
        """
        Reset the execution environment to its initial state
        """
        self.globals_dict = {}
        self.locals_dict = {}
        print("[DEBUG] IPython interpreter environment reset")
    
    def is_finish(self, output_str:str) -> bool:
        return self._is_finish


def execute_python_code(code, context: dict = dict()):
    interpreter = IPythonInterpreter(context)

    state = interpreter.execute(code)
    status = state.get("status")
    results = state.get("results")

    if status == "success":    
        results.pop("__name__")
        results.pop("_operations_count")
        print_outputs = results.get("_print_outputs")
        results.pop("_print_outputs")
        
        variables = results.copy()
        context.update(variables)

        output_str = format_variables(
            variables = variables, 
        )
    else:
        output_str = results
    
    return output_str, context


description = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Execute Python code in a sandboxed environment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    },
                },
                "required": ["code"],
            },
        },
    }
]



if __name__ == "__main__":
#     test_str = f"""
# <think>
# ...
# </think>


# <tool_call>    
# {{"name": "execute_python_code", "arguments": {{"code": "
# def find_pos_integers(limit):
#     results = {{}}
#     for x in range(1, limit+1):
#         for y in range(1, limit+1):
#             value = (x**2 + y) / (x*y + 1)
#             if value.is_integer() and value > 0:
#                 if value not in results:
#                     results[value] = [(x, y)]
#                 else:
#                     if len(results[value]) < 2:
#                         results[value].append((x, y))
#     return results
# "}}}}
# </tool_call>"""

#     def parse_tool_calls(content:str):
#         tool_calls = []
#         offset = 0
#         pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
#         for i, m in enumerate(pattern.finditer(content)):
#             if i == 0:
#                 offset = m.start()
#             func = json.loads(m.group(1).strip().replace("\n", "\\n"))
#             tool_calls.append({"type": "function", "function": func})
#             if isinstance(func["arguments"], str):
#                 func["arguments"] = json.loads(func["arguments"])
#         if tool_calls:
#             if offset > 0 and content[:offset].strip():
#                 c = content[:offset]
#             else: 
#                 c = ""
#             return {"role": "assistant", "content": c, "tool_calls": tool_calls}
#         return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}
    
#     print(parse_tool_calls(test_str))
#     code = parse_tool_calls(test_str)["tool_calls"][0]["function"]["arguments"]["code"]
#     print(execute_python_code(code))

    code = """\
def func():
    return"""
    print(execute_python_code(code))
