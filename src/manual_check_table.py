"""Table for the manual check of the automatic labels on the 200 uids of data/subsets/manual_check_uids.json.

Writes results/manual_check_200.xlsx. Column `auto_correct` is hidden so the manual label is set blind;
unhide it only after all 200 rows are filled. Fill `my_correct` with 1 (correct), 0 (wrong) or ? (cannot decide,
for example an incomplete reference), and add a comment for every ? and every disagreement.
Needs pymorphy3, openpyxl and RuBQ (RUBQ_JSON or the default path in sp_common).
"""
import json, os, sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import correct, match_lemma, q_kind
from sp_common import load_rubq

DATA = "data/main/generations_main.jsonl"
UIDS = "data/subsets/manual_check_uids.json"
OUT = "results/manual_check_200.xlsx"
COLS = [("uid", 7), ("kind", 8), ("question", 50), ("answer_text", 22), ("ru_names", 40), ("greedy", 25),
        ("auto_correct", 8), ("my_correct", 11), ("comment", 40)]


def ru_names(q, limit=12):
    out = []
    for a in q["answers"]:
        out += [a.get("label") or a["value"]] + a["wd_names"]["ru"] + a["wp_names"]
    return " | ".join(list(dict.fromkeys(s for s in out if s))[:limit])


def main():
    uids = json.load(open(UIDS, encoding="utf-8"))["uids"]
    rows = {}
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["uid"]] = r
    rubq = load_rubq()
    wb = Workbook()
    ws = wb.active
    ws.title = "manual_check"
    ws.append([c for c, _ in COLS])
    for u in uids:
        q, r = rubq[u], rows[u]
        ws.append([u, q_kind(q), q["question_text"], q["answer_text"], ru_names(q), r["greedy"],
                   int(correct(r["greedy"], q, match_lemma)), None, None])
    for i, (name, width) in enumerate(COLS, start=1):
        letter = ws.cell(row=1, column=i).column_letter
        ws.column_dimensions[letter].width = width
        ws.cell(row=1, column=i).font = Font(bold=True)
        if name == "auto_correct":
            ws.column_dimensions[letter].hidden = True
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wb.save(OUT)
    print(f"wrote {OUT}: {len(uids)} rows")


if __name__ == "__main__":
    main()
