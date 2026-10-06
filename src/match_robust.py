"""Robustness-only matching. Not used for the main figures; match.py stays frozen.

Differences from match_lemma: for number and date questions a month name counts as its number,
and two answers with equal cleaned text match even when no digits can be extracted
(«Август» and «август», «IX век» and «IX век»).
"""
from match import MONTH_RX, MONTHS, canon_number, clean, match_lemma, match_strict


def month_number(s):
    m = MONTH_RX.search(s.split("\n")[0].lower())
    if not m:
        return None
    return next(v for k, v in MONTHS.items() if m.group(1).startswith(k))


def match_robust(a, b, kind="entity"):
    if kind == "number":
        na, nb = canon_number(a), canon_number(b)
        na = month_number(a) if na is None else na
        nb = month_number(b) if nb is None else nb
        if na is not None and na == nb:
            return True
    elif kind == "date":
        if match_strict(a, b, kind):
            return True
    else:
        return match_lemma(a, b, kind)
    ca = clean(a)
    return bool(ca) and ca == clean(b)
