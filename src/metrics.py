"""Шаг 4. Метрики предпосылки. Использует match.py без изменений."""
import json, random, itertools, glob
from collections import Counter
import numpy as np
from match import match_strict, match_lemma, q_kind, gold_strings, lemmas, clean

def load(gen_path, rubq_path):
    rubq = {q["uid"]: q for q in json.load(open(rubq_path))}
    rows = [json.loads(l) for l in open(gen_path)]
    return rows, rubq

def correct(ans, q, fn):
    k = q_kind(q)
    return any(fn(ans, g, k) for g in gold_strings(q))

def group_sc(samples, kind, fn):
    """Жадная группировка: сэмпл идёт в первую группу, с представителем которой совпадает."""
    reps, sizes = [], []
    for s in samples:
        for i, r in enumerate(reps):
            if fn(s, r, kind):
                sizes[i] += 1
                break
        else:
            reps.append(s); sizes.append(1)
    return max(sizes) / len(samples)

def per_question(rows, rubq):
    out = []
    for r in rows:
        q = rubq[r["uid"]]; k = q_kind(q); S = r["samples"]
        pairs_l = pairs_ls = 0
        merged = []
        for i, j in itertools.combinations(range(len(S)), 2):
            l = match_lemma(S[i], S[j], k)
            if l:
                pairs_l += 1
                if not match_strict(S[i], S[j], k):
                    pairs_ls += 1; merged.append((S[i], S[j]))
        out.append(dict(
            uid=r["uid"], kind=k, question=r["question"], greedy=r["greedy"],
            gold=q["answers"][0].get("label") or q["answers"][0]["value"],
            ok_s=correct(r["greedy"], q, match_strict), ok_l=correct(r["greedy"], q, match_lemma),
            pairs_l=pairs_l, pairs_ls=pairs_ls, merged=merged,
            sc_s=group_sc(S, k, match_strict), sc_l=group_sc(S, k, match_lemma),
            mean_lp=float(np.mean(r["greedy_logprobs"])) if r["greedy_logprobs"] else float("nan"),
        ))
    return out

def stats(P):
    ok_s = np.array([p["ok_s"] for p in P]); ok_l = np.array([p["ok_l"] for p in P])
    any_ok = ok_s | ok_l; diff = ok_s != ok_l
    pl = np.array([p["pairs_l"] for p in P]); pls = np.array([p["pairs_ls"] for p in P])
    dsc = np.array([p["sc_l"] - p["sc_s"] for p in P])
    return {
        "Acc_strict": ok_s.mean(), "Acc_lemma": ok_l.mean(),
        "D1": diff.sum() / max(any_ok.sum(), 1),
        "D2": pls.sum() / max(pl.sum(), 1),
        "D2_sc": (np.abs(dsc) > 1e-9).mean(), "mean_abs_dSC": np.abs(dsc).mean(),
    }

def counts(P):
    ok_s = np.array([p["ok_s"] for p in P]); ok_l = np.array([p["ok_l"] for p in P])
    return {"n": len(P), "D1_abs": int((ok_s != ok_l).sum()), "any_ok": int((ok_s | ok_l).sum()),
            "strict_only": int((ok_s & ~ok_l).sum()), "lemma_only": int((ok_l & ~ok_s).sum()),
            "pairs_lemma": int(sum(p["pairs_l"] for p in P)),
            "pairs_lemma_not_strict": int(sum(p["pairs_ls"] for p in P)),
            "q_sc_changed": int(sum(abs(p["sc_l"] - p["sc_s"]) > 1e-9 for p in P))}

def bootstrap(P, B=1000, seed=42):
    rng = np.random.default_rng(seed); n = len(P)
    boot = [stats([P[i] for i in rng.integers(0, n, n)]) for _ in range(B)]
    return {k: (float(np.percentile([b[k] for b in boot], 2.5)),
                float(np.percentile([b[k] for b in boot], 97.5))) for k in boot[0]}

def decision(st, ci):
    """Правило решения; при пересечении порога интервалом берём более осторожный вариант (по нижним границам)."""
    if st["Acc_lemma"] < 0.15:
        return "Acc_lemma < 15%: switch to one larger model, repeat steps 2-4, no decision", None, False
    def rule(a, b):
        if a >= 0.10 or b >= 0.10: return "Premise strong: proceed with the thesis plan as written"
        if a < 0.03 and b < 0.03: return "Mechanism does not work: reformulate the topic or switch to the backup topic"
        return "Proceed with emphasis on H2 (sample grouping) and native data"
    point = rule(st["D1"], st["D2_sc"])
    cautious = rule(ci["D1"][0], ci["D2_sc"][0])
    # граница: 95% CI хотя бы одной метрики содержит порог 3% или 10%
    border = any(lo < t < hi for t in (0.03, 0.10) for (lo, hi) in (ci["D1"], ci["D2_sc"]))
    return point, cautious, border

def manual_sample(P, n=50, seed=42):
    """50 случайных случаев: D1-спасения (ответ vs эталон) + D2-слияния (сэмпл vs сэмпл)."""
    cases = []
    for p in P:
        if p["ok_l"] and not p["ok_s"]:
            cases.append(dict(src="D1", uid=p["uid"], question=p["question"], a=p["greedy"], b=p["gold"]))
        for a, b in dict.fromkeys(p["merged"]):   # уникальные пары строк
            cases.append(dict(src="D2", uid=p["uid"], question=p["question"], a=a, b=b))
    random.Random(seed).shuffle(cases)
    for c in cases:
        c["lemmas_a"] = " ".join(sorted(lemmas(c["a"]).elements()))
        c["lemmas_b"] = " ".join(sorted(lemmas(c["b"]).elements()))
        c["false_merge"] = ""   # заполнить вручную: 1 = разные сущности слиплись, 0 = корректно
    return cases[:n], len(cases)
