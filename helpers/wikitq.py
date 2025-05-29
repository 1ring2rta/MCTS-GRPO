import ast
import re
import unicodedata
from math import isnan, isinf

import json, os, pandas as pd
from datasets import Dataset


# ---------- copy from wikitq evaluator ----------
def normalize(x):
    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", '"', x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    if x and x[-1] == '.':
        x = x[:-1]
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x

class Value:
    def match(self, other): raise NotImplementedError
    @property
    def normalized(self): return self._normalized

class StringValue(Value):
    def __init__(self, content):
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)
    def __eq__(self, o): return isinstance(o, StringValue) and self.normalized == o.normalized
    def __hash__(self): return self._hash
    def __str__(self):  return 'S' + str([self.normalized])
    __repr__ = __str__
    def match(self, other): return self.normalized == other.normalized

class NumberValue(Value):
    def __init__(self, amount, original_string=None):
        self._amount = int(amount) if abs(amount-round(amount))<1e-6 else float(amount)
        self._normalized = normalize(original_string) if original_string else str(self._amount)
        self._hash = hash(self._amount)
    @property
    def amount(self): return self._amount
    def __eq__(self, o): return isinstance(o, NumberValue) and abs(self.amount-o.amount)<1e-6
    def __hash__(self): return self._hash
    def __str__(self):  return 'N(%f)'%self.amount + str([self.normalized])
    __repr__ = __str__
    def match(self, other):
        if self.normalized == other.normalized: return True
        return isinstance(other, NumberValue) and abs(self.amount-other.amount)<1e-6
    @staticmethod
    def parse(text):
        try:  return int(text)
        except:
            try:
                v=float(text); assert not isnan(v) and not isinf(v); return v
            except: return None

class DateValue(Value):
    def __init__(self, y,m,d,original_string=None):
        self._year,self._month,self._day = y,m,d
        self._normalized = normalize(original_string) if original_string else f'{y if y!=-1 else "xx"}-{m if m!=-1 else "xx"}-{d if d!=-1 else "xx"}'
        self._hash = hash((y,m,d))
    @property
    def ymd(self): return (self._year,self._month,self._day)
    def __eq__(self,o): return isinstance(o,DateValue) and self.ymd==o.ymd
    def __hash__(self): return self._hash
    def __str__(self):  return f'D{self.ymd}'+str([self._normalized])
    __repr__ = __str__
    def match(self, other):
        if self.normalized==other.normalized: return True
        return isinstance(other,DateValue) and self.ymd==other.ymd
    @staticmethod
    def parse(text):
        try:
            y,m,d = text.lower().split('-'); assert len([y,m,d])==3
            y = -1 if y in ('xx','xxxx') else int(y)
            m = -1 if m=='xx' else int(m)
            d = -1 if d=='xx' else int(d)
            assert not (y==m==d==-1)
            assert m==-1 or 1<=m<=12
            assert d==-1 or 1<=d<=31
            return (y,m,d)
        except: return None

# ---------- functional tools ----------
def to_value(original):
    """str -> Value"""
    amount = NumberValue.parse(original)
    if amount is not None: return NumberValue(amount, original)
    ymd = DateValue.parse(original)
    if ymd is not None:
        if ymd[1]==ymd[2]==-1:
            return NumberValue(ymd[0], original)   # 年份单独出现按数字
        return DateValue(*ymd, original_string=original)
    return StringValue(original)

def to_value_list(strs):
    return list({to_value(s) for s in strs})

def check_denotation(target_values, predicted_values):
    if len(target_values)!=len(predicted_values): return False
    return all(any(t.match(p) for p in predicted_values) for t in target_values)

def extract_answer(model_output):
    m = re.search(r'<answer>(.*?)</answer>', model_output, re.S | re.I)
    if not m:
        return []
    span = m.group(1).strip()
    try:
        items = ast.literal_eval(span)
        assert isinstance(items, (list, tuple))
        return [str(s) for s in items if str(s).strip()]
    except Exception:
        return [span] if span else []

# ---------- 4. reward ----------
def wikitq_reward(model_output, ground_truth):
    """
    Args
    ----
    model_output : str
        text that contains exactly one <answer>...</answer> block
    ground_truth : list[str]  (raw target strings)

    Returns
    -------
    float : 1.0 if match else 0.0
    """
    pred_values   = to_value_list(extract_answer(model_output))
    target_values = to_value_list(extract_answer(f"<answer>{ground_truth}</answer>"))
    return 1.0 if check_denotation(target_values, pred_values) else 0.0


# ------------------ data loader ------------------
def load_wikitq(jsonl_path: str) -> Dataset:
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
                    "question": q + "\n👆\nformat the final answer as `<answer>...</answer>`, the content within must be a list, e.g. `The capital of America` -> `<answer>['Washington']</answer>`",
                    "answer": str(a),
                    "table_path": [t_path],
                }
            )
    return Dataset.from_list(records)


# ------------------ Demo ------------------
if __name__ == "__main__":
    gt  = "['Paris', 'London']"
    out = "<answer>['London', 'Paris']</answer>"
    print(wikitq_reward(out, gt))
