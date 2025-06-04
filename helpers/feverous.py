from datasets import Dataset
import os, re, json


def extract_single_answer(model_output: str) -> str:
    """Extracts and returns the string inside the first <answer>…</answer> block, lowercased."""
    m = re.search(r'<answer>(.*?)</answer>', model_output, re.S | re.I)
    if not m:
        return ""
    return m.group(1).strip().lower()


def feverous_reward(model_output: str, ground_truth: str) -> float:
    """
    Args
    ----
    model_output : str
        Text that contains exactly one <answer>…</answer> block,
        e.g. "<answer>supports</answer>" or "<answer>refutes</answer>".
    ground_truth : str
        The correct label, either "supports" or "refutes".
    Returns
    -------
    float : 1.0 if the predicted label matches the ground truth (case-insensitive), else 0.0.
    """
    pred = extract_single_answer(model_output)
    gt   = ground_truth.strip().lower()
    return 1.0 if pred == gt else 0.0


# ------------------ data loader ------------------
def load_feverous(jsonl_path: str) -> Dataset:
    """Return HF Dataset with fields: prompt, ground_truth, table_path."""
    root = os.path.dirname(jsonl_path)
    tables_dir = os.path.join(root, "tables")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q  = obj["question"]
            a = obj["answers"]      
            t_id = obj["table_id"]
            t_path = os.path.join(tables_dir, f"{t_id}.csv")
            records.append(
                {
                    "question": q + "\n👆\nformat the final answer as `<answer>...</answer>`, the content is supports/refutes, e.g. `The capital of America is Washington` -> `<answer>supports</answer>`",
                    "answer": str(a),
                    "table_path": [t_path],
                }
            )
    return Dataset.from_list(records)


# ------------------ Demo ------------------
if __name__ == "__main__":
    gt  = "supports"
    out = "<answer>refutes</answer>"
    print(feverous_reward(out, gt))
