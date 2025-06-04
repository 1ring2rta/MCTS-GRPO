import json
from typing import Dict, List, Callable
from logging import getLogger
from smolagents import LocalPythonExecutor
from tools.tool_base import Tool
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import re
pa = r'(\b[\w\d\(\)\+\-\*/\s]+\b)\s*\*\*\s*(\d{6,}|(\(*\s*\d+\s*\**\s*\**\s*\d+\s*\)*))'

DEFAULT_VARS = {
}
logger = getLogger(__file__)


class IPythonInterpreter(Tool):
    """
    Tool for executing Python code similar to a Jupyter notebook
    """
    
    def __init__(self, local_dict):
        """ Initialize the IPython interpreter tool
        Args:
            timeout: int timeout for excuting a python code.
            init_funcs: Dict[Callable] init local varibales for excuting python codes.
        """
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
            additional_authorized_imports=[],
        )
        self.executor.send_tools({})
    
    def execute(self, code: str) -> dict:
        try:
            self.executor(code)
            return {"status": "success", "results": self.executor.state}
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

        output_str = f"PRINT_OUTPUTS:\n{print_outputs}\n"
        for k, v in variables.items():
            output_str += f"Var: {k}; Type: {type(v)}\n{v}\n"
    else:
        output_str = results
    
    return output_str, context


description = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Execute Python code in a sandboxed environment with timeout and capture output.",
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
    # interpreter = IPythonInterpreter(local_dict={})
    # print(interpreter.execute("result1 = 10\nresult2 = result1*8"))
    execute_python_code("result1 = 10\nresult2 = result1*8")
