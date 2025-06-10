import multiprocessing
import re


def exe_lambda(
    packages: str,
    lambda_expression: str,
    output_variable_name: str = "tmp_result", 
    timeout: int = 10,
    context: dict = None
):
    ns = {}
    if context:
        ns.update(context)
    ns['__builtins__'] = __builtins__

    exec(packages, ns)

    URL_PAT = re.compile(r"https?://|ftp://", re.I)
    if URL_PAT.search(lambda_expression):
        raise RuntimeError("Network access is disabled in offline mode.")

    fn = eval(lambda_expression, ns)

    def _run_lambda(f, q):
        try:
            q.put(f())
        except Exception as e:
            q.put(e)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_lambda, args=(fn, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        raise TimeoutError("Lambda execution timed out.")

    result = q.get()
    if isinstance(result, Exception):
        raise result

    output_str = f"Var: {output_variable_name}; Type: {type(result)}\n{result}"
    context[output_variable_name] = result
    return output_str, context


description = [
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
                        "description": "Import statements or setup code needed prior to execution, e.g. 'import pandas as pd; import numpy as np'."
                    },
                    "lambda_expression": {
                        "type": "string",
                        "description": "A valid Python lambda expression using only globals from the provided context, e.g. 'lambda: a + b'."
                    },
                    "output_variable_name": {
                        "type": "string",
                        "description": "Name of the variable under which to store the lambda result in the context; defaults to 'tmp_result' if omitted."
                    }
                },
                "required": [
                    "lambda_expression"
                ]
            }
        }
    }
]
