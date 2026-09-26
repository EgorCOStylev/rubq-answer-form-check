"""Шаг 3. Функции сопоставления. Фиксируются ДО подсчёта метрик.

match_strict(a, b, kind) и match_lemma(a, b, kind) применяются без изменений
и к парам (ответ, эталон/алиас), и к парам (сэмпл_i, сэмпл_j).
kind — тип эталона вопроса: 'entity' | 'string' | 'number' | 'date'.
"""
import re
import unicodedata
from collections import Counter

# --- общая предобработка (одинаковая для обеих функций) ---
def clean(s):
    s = s.split("\n")[0]                                   # только первая строка ответа
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # ударения: «Но́вая» → «Новая»
    s = unicodedata.normalize("NFC", s).lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s)                         # пунктуация → пробел
    return re.sub(r"\s+", " ", s).strip()

# --- числа и даты: одно простое правило, общее для strict и lemma ---
MONTHS = {  # основа месяца → номер; ловит «февраль», «февраля», «феврале»
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
MONTH_RX = re.compile(r"\b(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]|июн\w*|июл\w*|"
                      r"август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\b")

def canon_number(s):
    s = s.split("\n")[0].lower().replace(",", ".")
    s = re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", s)     # «1 500» → «1500»
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    x = float(m.group())
    return int(x) if x == int(x) else x

def canon_date(s):
    """→ (год, месяц, день); отсутствующие компоненты = None."""
    s = s.split("\n")[0].lower()
    iso = re.match(r"^\s*(-?\d{1,4})-(\d{2})-(\d{2})t", s)
    if iso:  # эталон Wikidata: '1860-07-02T00:00:00Z'
        return int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    year = re.search(r"\b(\d{3,4})\b", s)
    mon = MONTH_RX.search(s)
    month = None
    if mon:
        w = mon.group(1)
        month = next(v for k, v in MONTHS.items() if w.startswith(k))
    day = re.search(r"\b(\d{1,2})\b", s) if mon else None
    return (int(year.group(1)) if year else None, month, int(day.group(1)) if day else None)

def match_date(a, b):
    """Совпадение, если год есть у обоих и равен, а остальные компоненты
    не противоречат (указанные у обоих — равны). «1860» ≡ «1860-07-02»."""
    da, db = canon_date(a), canon_date(b)
    if da[0] is None or db[0] is None:
        return False
    return all(x is None or y is None or x == y for x, y in zip(da, db))

def match_number(a, b):
    na, nb = canon_number(a), canon_number(b)
    return na is not None and na == nb

# --- строгое ---
def match_strict(a, b, kind="entity"):
    if kind == "number":
        return match_number(a, b)
    if kind == "date":
        return match_date(a, b)
    ca, cb = clean(a), clean(b)
    return bool(ca) and ca == cb

# --- лемматизированное ---
STOP = {  # предлоги и союзы — фиксированный список
    "в", "во", "на", "из", "с", "со", "у", "о", "об", "обо", "по", "за", "под", "при", "от",
    "до", "к", "ко", "для", "через", "над", "перед", "между", "около", "без", "про",
    "и", "или", "а", "но", "да",
}
_morph = None
def _lemma(w):
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph.parse(w)[0].normal_form.replace("ё", "е")

def lemmas(s):
    return Counter(_lemma(t) for t in clean(s).split() if t not in STOP)

def match_lemma(a, b, kind="entity"):
    if kind in ("number", "date"):
        return match_strict(a, b, kind)       # числа/даты: то же правило, лемматизация не влияет
    la, lb = lemmas(a), lemmas(b)
    return bool(la) and la == lb

# --- эталон вопроса ---
DT = "http://www.w3.org/2001/XMLSchema#"
def q_kind(q):
    a = q["answers"][0]
    if a["type"] == "uri":
        return "entity"
    dt = a.get("datatype", "")
    if dt == DT + "dateTime":
        return "date"
    if dt in (DT + "integer", DT + "decimal"):
        return "number"
    return "string"

def gold_strings(q):
    """label/value + wd_names.ru + wd_names.en + wp_names по всем эталонным ответам."""
    out = []
    for a in q["answers"]:
        out += [a.get("label") or a["value"]]
        out += a["wd_names"]["ru"] + a["wd_names"]["en"] + a["wp_names"]
    return [s for s in dict.fromkeys(out) if s]

def correct(answer, q, fn):
    k = q_kind(q)
    return any(fn(answer, g, k) for g in gold_strings(q))
