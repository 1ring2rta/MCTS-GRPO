import os, json, re, string, collections, ast
from datasets import Dataset

# ---------- Text normalization ---------- #
def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def _exact_match(gold: str, pred: str) -> int:
    return int(_normalize_answer(gold) == _normalize_answer(pred))

# ---------- Extract <answer>...</answer> ---------- #
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)

def _extract_answer(span: str) -> str:
    m = _ANSWER_RE.search(span)
    return m.group(1).strip() if m else ""

# ---------- HybridQA Reward ---------- #
def hybridqa_reward(model_output: str, ground_truth):
    pred   = _extract_answer(model_output)
    golds  = ground_truth if isinstance(ground_truth, (list, tuple)) else [ground_truth]
    return 1.0 if any(_exact_match(g, pred) for g in golds) else 0.0

# ---------- Data Loader ---------- #
def _parse_answers(raw):
    """
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw]

    raw = str(raw).strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass

    for sep in ["|", ";", ","]:
        if sep in raw:
            return [x.strip() for x in raw.split(sep) if x.strip()]
    return [raw]

def load_hybridqa(jsonl_path: str) -> Dataset:
    root        = os.path.dirname(jsonl_path)
    tables_dir  = os.path.join(root, "tables")
    records     = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj   = json.loads(line)
            ques  = obj["question"]
            ans   = _parse_answers(obj["answers"])
            t_id  = obj["table_id"]
            t_csv = os.path.join(tables_dir, f"{t_id}.csv")

            ques = f"{ques}\n👆\nformat the final answer as `<answer>...</answer>`, e.g. `Ghurids` -> `<answer>Ghurids</answer>`"
            
            records.append({"question": ques, "answer": str(ans), "table_path": [t_csv]})

    return Dataset.from_list(records)

# ------------------ Demo ------------------ #
if __name__ == "__main__":
    gt  = ["Ghurids"]
    out = "<answer>Ghurids</answer>"
    print("reward =", hybridqa_reward(out, gt))  # 1.0

    out2 = "<answer>gurids</answer>"
    print("reward =", hybridqa_reward(out2, gt)) # 0.0
