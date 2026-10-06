import argparse, csv, itertools, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import match_lemma, q_kind, gold_strings
from metrics import correct
from sp_common import load_rubq, load_jsonl, degenerate_flags, MAX_NEW_TOKENS
from nli_core import (VARIANTS, unique_texts, same_nli, greedy_classes, se_frequency, pair_type)

B = 2000


def question_stats(r, q, texts, M, fn, with_ok=True):
    S, kind = r["samples"], q_kind(q)
    expected, idx = unique_texts(S)
    if expected != texts:
        raise ValueError(f"uid {r['uid']}: stored NLI texts do not match the samples")
    degen = [any(degenerate_flags(s, n >= MAX_NEW_TOKENS, q).values()) for s, n in zip(S, r["samples_ntok"])]
    nli_labels = greedy_classes(len(S), lambda a, b: same_nli(M, idx[a], idx[b]))
    lemma_labels = greedy_classes(len(S), lambda a, b: fn(S[a], S[b], kind))
    pairs = []
    for i, j in itertools.combinations(range(len(S)), 2):
        ls = bool(fn(S[i], S[j], kind))
        ns = same_nli(M, idx[i], idx[j])
        t = pair_type(S[i], S[j], ls, ns, degen[i] or degen[j])
        if t:
            pairs.append(dict(a=S[i], b=S[j], lemma_same=ls, nli_same=ns, type=t))
    return dict(uid=r["uid"], kind=kind, n_pairs=len(S) * (len(S) - 1) // 2, disagree=pairs,
                lemma_classes=len(set(lemma_labels)), nli_classes=len(set(nli_labels)),
                se=se_frequency(nli_labels), ok=bool(correct(r["greedy"], q, fn)) if with_ok else None)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def share_ci(stats, rng):
    d = np.array([len(s["disagree"]) for s in stats], float)
    n = np.array([s["n_pairs"] for s in stats], float)
    boot = [d[i].sum() / n[i].sum() for i in (rng.integers(0, len(d), len(d)) for _ in range(B))]
    return float(d.sum() / n.sum()), ci(boot)


def summarize(stats, rng):
    share, share_ci_ = share_ci(stats, rng)
    types = Counter(p["type"] for s in stats for p in s["disagree"])
    by_kind = {}
    for k in sorted({s["kind"] for s in stats}):
        sub = [s for s in stats if s["kind"] == k]
        by_kind[k] = dict(n=len(sub), share=sum(len(s["disagree"]) for s in sub) / sum(s["n_pairs"] for s in sub))
    se = np.array([s["se"] for s in stats])
    ok = np.array([s["ok"] for s in stats]) if stats[0]["ok"] is not None else None
    return dict(
        pairs=int(sum(s["n_pairs"] for s in stats)), disagree=int(sum(len(s["disagree"]) for s in stats)),
        share=share, share_ci=share_ci_,
        lemma_same_nli_split=int(sum(v for k, v in types.items() if k.startswith("split"))),
        nli_same_lemma_split=int(sum(v for k, v in types.items() if k.startswith("merge"))),
        types=dict(types), by_kind=by_kind,
        questions_with_disagreement=int(sum(bool(s["disagree"]) for s in stats)),
        questions_class_count_differs=int(sum(s["lemma_classes"] != s["nli_classes"] for s in stats)),
        se_finite=int(np.isfinite(se).sum()), se_zero=int((se == 0).sum()),
        se_mean_greedy_ok=float(se[ok].mean()) if ok is not None and ok.any() else None,
        se_mean_greedy_wrong=float(se[~ok].mean()) if ok is not None and (~ok).any() else None)


def paired_diff(stats_a, stats_b, rng):
    da = np.array([len(s["disagree"]) / s["n_pairs"] for s in stats_a])
    db = np.array([len(s["disagree"]) / s["n_pairs"] for s in stats_b])
    n = len(da)
    boot = [(db[i] - da[i]).mean() for i in (rng.integers(0, n, n) for _ in range(B))]
    return float((db - da).mean()), ci(boot)


def write_review(path, stats, rows, rubq, top=50):
    by_uid = {r["uid"]: r for r in rows}
    order = sorted(stats, key=lambda s: (-len(s["disagree"]), s["uid"]))[:top]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "question", "gold", "n_disagree", "lemma_classes", "nli_classes", "samples", "types"])
        for s in order:
            r = by_uid[s["uid"]]
            w.writerow([s["uid"], r["question"], "; ".join(gold_strings(rubq[s["uid"]])[:3]), len(s["disagree"]),
                        s["lemma_classes"], s["nli_classes"], " | ".join(x.split("\n")[0] for x in r["samples"]),
                        ", ".join(sorted({p["type"] for p in s["disagree"]}))])


def examples(stats, rows, k=10):
    by_uid = {r["uid"]: r for r in rows}
    out = []
    for s in sorted(stats, key=lambda s: (-len(s["disagree"]), s["uid"])):
        if s["disagree"]:
            p = s["disagree"][0]
            out.append(dict(uid=s["uid"], question=by_uid[s["uid"]]["question"], **p))
        if len(out) == k:
            break
    return out


def load_variant(v, uids, nli_dir="results"):
    path = f"{nli_dir}/nli_{v}.jsonl"
    data = {r["uid"]: r for r in load_jsonl(path)}
    missing = [u for u in uids if u not in data]
    if missing:
        sys.exit(f"{path} is incomplete: {len(missing)} of {len(uids)} questions missing; rerun src/run_nli.py")
    return data


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/generations.jsonl")
    ap.add_argument("--nli-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--no-correctness", action="store_true", help="do not evaluate greedy correctness (pre-AUROC checks)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args([] if argv is None else argv)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_jsonl(args.data)
    rubq = load_rubq()
    uids = [r["uid"] for r in rows]
    rng = np.random.default_rng(0)
    res, all_stats = {}, {}
    for v in args.variants:
        data = load_variant(v, uids, args.nli_dir)
        stats = [question_stats(r, rubq[r["uid"]], data[r["uid"]]["texts"], data[r["uid"]]["M"], match_lemma,
                          with_ok=not args.no_correctness)
                 for r in rows]
        all_stats[v] = stats
        res[v] = summarize(stats, rng)
        write_review(f"{args.out_dir}/review_top50_{v}.csv", stats, rows, rubq)
        res[v]["examples"] = examples(stats, rows)
        print(v, {k: x for k, x in res[v].items() if k != "examples"}, flush=True)
    if {"bare", "qa"} <= set(args.variants):
        mean, ci_ = paired_diff(all_stats["bare"], all_stats["qa"], rng)
        res["qa_minus_bare"] = dict(mean=mean, ci=ci_, ci_contains_zero=ci_[0] <= 0 <= ci_[1])
        print("qa_minus_bare", res["qa_minus_bare"], flush=True)
    json.dump(res, open(f"{args.out_dir}/nli_agreement.json", "w"), ensure_ascii=False, indent=1)
    return res, all_stats


if __name__ == "__main__":
    main(sys.argv[1:])
