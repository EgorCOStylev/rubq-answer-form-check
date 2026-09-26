"""Шаг 1: описательная статистика RuBQ 2.0 без модели."""
import json, re
from collections import Counter

DT = "http://www.w3.org/2001/XMLSchema#"

def answer_type(a):
    if a["type"] == "uri":
        return "entity"
    dt = a.get("datatype", "")
    if dt == DT + "dateTime":
        return "date"
    if dt in (DT + "integer", DT + "decimal"):
        return "number"
    return "string"

def q_type(q):
    ts = Counter(answer_type(a) for a in q["answers"])
    return ts.most_common(1)[0][0] if len(ts) == 1 else "mixed:" + "+".join(sorted(ts))

def ru_aliases(a):
    label = (a.get("label") or "").lower()
    names = {n.lower() for n in a["wd_names"]["ru"] + a["wp_names"]}
    names.discard(label)
    return names

# Шаблоны «косвенного» вопроса — фиксированы здесь, в отчёт идут как есть.
PREP = r"(?:в|во|на|из|с|со|у|о|об|обо|по|за|под|при|от|до|к|ко|для|через|над|перед|между|около|после|вокруг|против)"
WH = r"(?:как\w*|кем|ком|кого|кому|чем|чём|чего|чему|чей|чь\w*|котор\w*|сколь\w*)"
PATTERNS = [
    ("P1 предлог + вопр. слово (в каком, у кого, с кем, из какого, на каком…)", rf"(?:^|\s){PREP}\s+{WH}\b"),
    ("P2 где / куда / откуда", r"(?:^|\s)(?:где|куда|откуда)\b"),
    ("P3 кем / чем / кому / чему / кого / чего без предлога", r"(?:^|[\s,])(?:кем|чем|кому|чему|кого|чего)\b"),
    ("P4 косвенные формы «какой» без предлога (каким, какому, какую, каких, какими, каком)",
     r"(?:^|[\s,])(?:каким|какому|какую|каких|какими|каком)\b"),
    ("P5 «какой/какого/какой стране…» — неоднозначно (им./род./дат./тв./пр.)", r"(?:^|[\s,])(?:какой|какого|какое|какая)\b"),
]
STRONG = [p for p in PATTERNS if not p[0].startswith("P5")]

def match_any(text, pats):
    t = text.lower().replace("ё", "е")
    return [name for name, rx in pats if re.search(rx, t)]

def report(split):
    d = json.load(open(f"RuBQ/RuBQ_2.0/RuBQ_2.0_{split}.json"))
    ans = [q for q in d if q["answers"]]
    print(f"\n=== {split}: всего {len(d)}, с ответом {len(ans)}, без ответа (answers == []) {len(d)-len(ans)}")
    tag_na = sum("no_answer" in q["tags"] or "no-answer" in q["tags"] for q in d)
    print(f"   тег no_answer в tags: {tag_na}; все теги: {Counter(t for q in d for t in q['tags']).most_common()}")
    print("   типы ответов (по вопросу):", Counter(q_type(q) for q in ans).most_common())
    print("   вопросов с >1 эталонным ответом:", sum(len(q["answers"]) > 1 for q in ans))
    # алиасы
    n_with, counts = 0, []
    for q in ans:
        al = set().union(*(ru_aliases(a) for a in q["answers"]))
        counts.append(len(al))
        n_with += bool(al)
    wp_nonempty = sum(any(a["wp_names"] for a in q["answers"]) for q in ans)
    print(f"   вопросов с ≥1 русским алиасом (wd_names.ru ∪ wp_names, кроме label): {n_with}/{len(ans)} = {n_with/len(ans):.1%}")
    print(f"   алиасов на вопрос: среднее {sum(counts)/len(counts):.2f}, среди имеющих {sum(counts)/max(n_with,1):.2f}, медиана {sorted(counts)[len(counts)//2]}")
    print(f"   вопросов с непустым wp_names: {wp_nonempty}/{len(ans)}")
    # эвристика
    hits = Counter()
    strong = amb_only = 0
    for q in ans:
        m = match_any(q["question_text"], PATTERNS)
        hits.update(m)
        s = match_any(q["question_text"], STRONG)
        strong += bool(s)
        amb_only += (not s) and bool(m)
    for name, _ in PATTERNS:
        print(f"   {hits[name]:5d}  {name}")
    print(f"   ИТОГО сильные шаблоны P1–P4: {strong}/{len(ans)} = {strong/len(ans):.1%}; "
          f"+ только P5: {amb_only} → верхняя граница {(strong+amb_only)/len(ans):.1%}")
    return ans

if __name__ == "__main__":
    test = report("test")
    report("dev")
    import random
    random.seed(0)
    print("\nПримеры срабатываний (test):")
    for name, rx in PATTERNS:
        ex = [q["question_text"] for q in test if match_any(q["question_text"], [(name, rx)])]
        print(" ", name.split()[0], "|", " || ".join(random.sample(ex, min(3, len(ex)))))
