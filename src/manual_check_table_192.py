"""Table for the blind manual check of 192 labels on 100 fresh uids.

Writes results/manual_check_192_100.xlsx. Column `auto_correct` is hidden so the
manual label is set blind; unhide it only after all 100 rows are filled.
Fill `my_correct` with 1 (correct), 0 (wrong) or ? (cannot decide), and add a
comment for every ? and every disagreement.

Needs pymorphy3, openpyxl and RuBQ (RUBQ_JSON or the default path in sp_common).
Run on Kaggle where all dependencies are present.
"""
import json, os, sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_192 import correct_v2
from sp_common import load_rubq

DATA = "data/main/generations_main.jsonl"
UIDS = "data/subsets/manual_check_192_uids.json"
OUT = "results/manual_check_192_100.xlsx"
COLS = [("uid", 7), ("kind", 8), ("question", 50), ("answer_text", 22),
        ("ru_names", 40), ("greedy", 25), ("auto_correct", 8),
        ("my_correct", 11), ("comment", 40)]


def q_kind(q):
    t = q.get("answer_type", "entity")
    return "number" if t == "number" else "date" if t == "date" else "entity"


def ru_names(q, limit=12):
    out = []
    for a in q["answers"]:
        out += [a.get("label") or a["value"]] + a["wd_names"]["ru"] + a.get("wp_names", [])
    return " | ".join(list(dict.fromkeys(s for s in out if s))[:limit])


def refs_of(q):
    refs = ru_names(q).split(" | ")
    at = q.get("answer_text", "").strip()
    if at and at not in refs and at.lower() != "none":
        refs.append(at)
    return refs


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
    ws.title = "manual_check_v2"
    ws.append([c for c, _ in COLS])
    for u in uids:
        q, r = rubq[u], rows[u]
        kind = q_kind(q)
        refs = refs_of(q)
        auto = int(correct_v2(r["greedy"], refs, kind))
        ws.append([u, kind, q["question_text"], q.get("answer_text", ""),
                   ru_names(q), r["greedy"], auto, None, None])
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
