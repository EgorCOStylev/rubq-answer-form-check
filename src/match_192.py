"""Matching rule (main) and soft variant.

Fixed in a commit BEFORE recomputing the main run (Borisov, 2026-11-17).
Changes vs match.py v1:
  1. Reference set adds RuBQ answer_text.
  2. clean() maps Latin homoglyphs to Cyrillic ("Тennis" -> "теннис", "Бonn" -> "бонн").
  3. Dates: placeholder refs (Y,1,1) match a bare year; a true full ref requires
     full match; century words ("IX век") are errors in the main variant
     (less-precise = error, Borisov decision D-028), granted in the soft variant.
Numbers keep the v1 rule in the main variant; the soft variant adds +-0.5% tolerance
and token-containment.
NOTE: v1 lemma multiset uses pymorphy3 (Kaggle). This preview uses a simple suffix
stripper as a sandbox fallback; final numbers must be confirmed on Kaggle with
pymorphy3 (same semantics as match.py v1).
"""
import re
import unicodedata
from collections import Counter

# Latin -> Cyrillic homoglyphs (main-rule component)
_HOMOGLYPH_MAP = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    "t": "т", "h": "н", "b": "в", "k": "к", "m": "м", "i": "и", "s": "с",
    "n": "н", "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С", "X": "Х",
    "Y": "У", "T": "Т", "H": "Н", "B": "В", "K": "К", "M": "М", "I": "И",
    "S": "С", "N": "Н",
}
HOMOGLYPH = str.maketrans(_HOMOGLYPH_MAP)

MONTHS = {"январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
          "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12}
MONTH_RX = re.compile(r"\b(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]|июн\w*|июл\w*|"
                      r"август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\b")
ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
CENTURY_RX = re.compile(r"^([ivxlcdm]+)\s*век")

STOP = {"в", "во", "на", "из", "с", "со", "у", "о", "об", "обо", "по", "за", "под",
        "при", "от", "до", "к", "ко", "для", "через", "над", "перед", "между",
        "около", "без", "про", "и", "или", "а", "но", "да"}

ALT_HOMOGLYPH = str.maketrans({**_HOMOGLYPH_MAP, "c": "к", "C": "К"})

def clean(s, alt=False):
    """alt=True: Latin 'c' maps to Cyrillic 'к' (covers 'Nicotin' -> 'никотин')."""
    s = s.split("\n")[0]
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    table = ALT_HOMOGLYPH if alt else HOMOGLYPH
    s = unicodedata.normalize("NFC", s).translate(table).lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def canon_number(s):
    s = s.split("\n")[0].lower().replace(",", ".")
    s = re.sub(r"(?<=\d)[\s\u00a0](?=\d{3}\b)", "", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    x = float(m.group())
    return int(x) if x == int(x) else x

def canon_date(s):
    s = s.split("\n")[0].lower()
    iso = re.match(r"^\s*(-?\d{1,4})-(\d{2})-(\d{2})(?:t|$|\s)", s)
    if iso:
        return int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    year = re.search(r"\b(\d{3,4})\b", s)
    mon = MONTH_RX.search(s)
    month = None
    if mon:
        w = mon.group(1)
        month = next(v for k, v in MONTHS.items() if w.startswith(k))
    day = re.search(r"\b(\d{1,2})\b", s) if mon else None
    return (int(year.group(1)) if year else None, month, int(day.group(1)) if day else None)

def century_of(s):
    s = unicodedata.normalize("NFC", s.split("\n")[0]).lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = CENTURY_RX.match(s)
    if not m:
        return None
    vals = [ROMAN[ch] for ch in m.group(1) if ch in ROMAN]
    if len(vals) != len(m.group(1)):
        return None
    total = 0
    for i, v in enumerate(vals):
        if i + 1 < len(vals) and vals[i + 1] > v:
            total -= v
        else:
            total += v
    return total

def match_date_main(a, b):
    """Main: years equal; every component of the answer must equal the ref;
    placeholder refs (Y,1,1) match a bare year; true full refs require full match."""
    da, db = canon_date(a), canon_date(b)
    if da[0] is None or db[0] is None or da[0] != db[0]:
        return False
    ay, am, ad = da
    by, bm, bd = db
    if am is None and ad is None:              # answer gives year only
        return True                            # placeholder (Y,1,1) or year-only answer
    if bm is None:
        return False                           # ref year-only, answer gives more -> main: error
    if am != bm:
        return False
    if ad is None:
        return bd is None or (bm == 1 and bd == 1)   # treat (Y,1,1) as year-only ref
    return ad == bd

def match_number_main(a, b):
    na, nb = canon_number(a), canon_number(b)
    return na is not None and na == nb

def _bare_year(s, n):
    letters = re.sub(r"[\d\s\W_]+", "", s, flags=re.UNICODE)
    return not letters and isinstance(n, int) and 1000 <= abs(n) <= 2100

def match_number_soft(a, b):
    """Rounding tolerance for quantities; bare years stay strict (wd is truth;
    off-by-one years are a known failure mode)."""
    na, nb = canon_number(a), canon_number(b)
    if na is None or nb is None:
        return False
    if na == nb:
        return True
    if _bare_year(a, na) and isinstance(nb, int) and 1000 <= abs(nb) <= 2100:
        return False
    return abs(na - nb) <= max(1, 0.005 * abs(nb))

def match_date_soft(a, b):
    if match_date_main(a, b):
        return True
    da, db = canon_date(a), canon_date(b)
    if da[0] is not None and da[0] == db[0] and da[1] is None and da[2] is None:
        return True  # year-only answer vs a full ref: soft grant (D-028)
    ca, cb = century_of(a), century_of(b)
    if ca is not None:
        by = canon_date(b)[0]
        return by is not None and (by - 1) // 100 + 1 == ca
    cb = century_of(b)
    if cb is not None:
        ay = canon_date(a)[0]
        return ay is not None and (ay - 1) // 100 + 1 == cb
    return False

_morph = None
def _lemma(w):
    global _morph
    if _morph is None:
        try:
            import pymorphy3
            _morph = pymorphy3.MorphAnalyzer()
        except ImportError:
            _morph = False
    if _morph:
        return _morph.parse(w)[0].normal_form.replace("ё", "е")
    return _stem_fallback(w)

_ENDINGS = ("овой", "евой", "ого", "евого", "их", "ых", "ыми", "ими", "ов", "ев",
            "ом", "ем", "ами", "ями", "ая", "яя", "ое", "ее", "ые", "ие", "ой",
            "ый", "ий", "ую", "юю", "у", "ю", "е", "а", "я", "о", "ы", "и", "ь")

def _stem_fallback(w):
    if len(w) <= 4:
        return w
    for end in sorted(_ENDINGS, key=len, reverse=True):
        if w.endswith(end) and len(w) - len(end) >= 3:
            return w[: -len(end)]
    return w

def lemmas(s):
    return Counter(_lemma(t) for t in clean(s).split() if t not in STOP)

def match_text_main(a, b):
    for alt in (False, True):
        ca, cb = clean(a, alt), clean(b, alt)
        if ca and (ca == cb or lemmas(a) == lemmas(b)):
            return True
    return False

def match_text_soft(a, b):
    if match_text_main(a, b):
        return True
    la, lb = lemmas(a), lemmas(b)
    return bool(la) and bool(lb) and all(lb[t] >= c for t, c in la.items())

def correct_v2(answer, refs, kind, soft=False):
    """refs: gold_strings(q) + [RuBQ answer_text]."""
    if kind == "number":
        return any(match_number_soft(answer, r) if soft else match_number_main(answer, r) for r in refs)
    if kind == "date":
        return any(match_date_soft(answer, r) if soft else match_date_main(answer, r) for r in refs)
    return any(match_text_soft(answer, r) if soft else match_text_main(answer, r) for r in refs)
