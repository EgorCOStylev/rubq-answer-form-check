"""Table for the manual check of the automatic labels on the 200 uids of data/subsets/manual_check_uids.json.

Fill `my_correct` with 1 (the greedy answer is correct), 0 (wrong) or ? (cannot decide), and add a comment
where you disagree with `auto_correct`. Needs pymorphy3 and RuBQ (RUBQ_JSON or the default path in sp_common).
"""
import csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import correct, match_lemma, q_kind
from sp_common import load_jsonl, load_rubq

DATA = "data/main/generations_main.jsonl"
UIDS = "data/subsets/manual_check_uids.json"
OUT = "results/manual_check_200.csv"


def ru_names(q, limit=12):
    out = []
    for a in q["answers"]:
        out += [a.get("label") or a["value"]] + a["wd_names"]["ru"] + a["wp_names"]
    return " | ".join(list(dict.fromkeys(s for s in out if s))[:limit])


def main():
    uids = json.load(open(UIDS))["uids"]
    rows = {r["uid"]: r for r in load_jsonl(DATA)}
    rubq = load_rubq()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "kind", "question", "answer_text", "ru_names", "greedy", "auto_correct", "my_correct", "comment"])
        for u in uids:
            q, r = rubq[u], rows[u]
            w.writerow([u, q_kind(q), q["question_text"], q["answer_text"], ru_names(q), r["greedy"],
                        int(correct(r["greedy"], q, match_lemma)), "", ""])
    print(f"wrote {OUT}: {len(uids)} rows")


if __name__ == "__main__":
    main()
