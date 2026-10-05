import argparse, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import match_strict, match_lemma, q_kind
from metrics import correct
from sp_common import load_rubq, load_jsonl
from signals import sc_uncertainty, auroc, aurc_tie_aware, aurc_naive, aurc_bounds, aurc_ideal


def bootstrap(u, err, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(u)
    a, r = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        a.append(aurc_tie_aware(u[i], err[i]))
        r.append(auroc(u[i], err[i]))
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return ci(a), ci(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", choices=["lemma", "strict"], default="lemma")
    args = ap.parse_args()
    fn = match_lemma if args.match == "lemma" else match_strict
    rows = load_jsonl("data/generations.jsonl")
    if not rows:
        sys.exit("data/generations.jsonl is missing or empty")
    rubq = load_rubq()
    per = []
    for r in rows:
        q = rubq[r["uid"]]
        per.append(dict(uid=r["uid"], sc=sc_uncertainty(r["samples"], q_kind(q), fn),
                        err=int(not correct(r["greedy"], q, fn))))
    u = np.array([p["sc"] for p in per])
    err = np.array([p["err"] for p in per])
    aurc_ci, auroc_ci = bootstrap(u, err)
    best, worst = aurc_bounds(u, err)
    res = dict(
        match=args.match, n=len(per), error_rate=float(err.mean()),
        distinct_values=int(len(np.unique(u))),
        value_counts={str(k): v for k, v in sorted(Counter(u.tolist()).items())},
        aurc_tie_aware=aurc_tie_aware(u, err), aurc_tie_aware_ci=aurc_ci,
        aurc_naive=aurc_naive(u, err), aurc_best_order=best, aurc_worst_order=worst,
        aurc_ideal=aurc_ideal(err), aurc_random=float(err.mean()),
        auroc_error=auroc(u, err), auroc_error_ci=auroc_ci, per_question=per)
    out = "results/sc.json" if args.match == "lemma" else "results/sc_strict.json"
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print({k: v for k, v in res.items() if k not in ("per_question", "value_counts")}, flush=True)


if __name__ == "__main__":
    main()
