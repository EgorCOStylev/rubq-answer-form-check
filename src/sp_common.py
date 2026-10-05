import json, os, re
from match import gold_strings

SUBSET_SEED = 20261004
N_SUBSET = 100
BASE_SEED = 42
MAX_NEW_TOKENS = 16
N_SAMPLES = 10
DEGEN_MARGIN = 0.02
TIE_PP = 0.01

CONFIGS = {
    "A": dict(temperature=1.0, top_k=0, top_p=1.0),
    "B": dict(temperature=0.7, top_k=0, top_p=0.95),
    "C": dict(temperature=1.0, top_k=50, top_p=0.95),
    "D": dict(temperature=0.8, top_k=50, top_p=0.95),
}
GENERATED = ["B", "C", "D"]

CJK_RX = re.compile("[㐀-䶿一-鿿豈-﫿　-〿぀-ヿ가-힯＀-￯]")
CYR_RX = re.compile("[А-Яа-яЁё]")


def has_cjk(t):
    return bool(CJK_RX.search(t))


def cyr_share(t):
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return None
    return sum(bool(CYR_RX.match(c)) for c in letters) / len(letters)


def gold_has_cyr(q):
    return any(CYR_RX.search(g) for g in gold_strings(q))


def degenerate_flags(text, hit_max, q):
    sh = cyr_share(text)
    return {
        "a": bool(hit_max),
        "b": has_cjk(text),
        "c": bool(gold_has_cyr(q) and sh is not None and sh < 0.5),
    }


def load_rubq(path=None):
    path = path or os.environ.get("RUBQ_JSON", "RuBQ/RuBQ_2.0/RuBQ_2.0_test.json")
    return {q["uid"]: q for q in json.load(open(path))}


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def subset_ids(path="data/sampling_params/question_ids.json"):
    return json.load(open(path))["uids"]
