import argparse, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import match_strict, match_lemma, q_kind, gold_strings
from sp_common import (CONFIGS, MAX_NEW_TOKENS, N_SAMPLES, DEGEN_MARGIN, TIE_PP,
                       subset_ids, load_jsonl, load_rubq, degenerate_flags)


def correct(ans, q, fn):
    k = q_kind(q)
    return any(fn(ans, g, k) for g in gold_strings(q))


def group_count(samples, kind):
    reps = []
    for s in samples:
        if not any(match_strict(s, r, kind) for r in reps):
            reps.append(s)
    return len(reps)


def load_cfg_rows(cfg, uids):
    if cfg == "A":
        rows = {r["uid"]: r for r in load_jsonl("data/generations.jsonl")}
        return {u: dict(samples=rows[u]["samples"], ntok=rows[u]["samples_ntok"], t=None, raw=None) for u in uids}
    path = os.environ.get("SP_OUT", "data/sampling_params/generations_sp.jsonl")
    rows = {r["uid"]: r for r in load_jsonl(path) if r["cfg"] == cfg}
    return {u: dict(samples=rows[u]["samples"], ntok=rows[u]["samples_ntok"], t=rows[u]["t_sec"],
                    raw=rows[u]["samples_raw"]) for u in uids if u in rows}


def per_question(rows, rubq, fn):
    out = {}
    for u, r in rows.items():
        q = rubq[u]
        flags = [degenerate_flags(s, n >= MAX_NEW_TOKENS, q) for s, n in zip(r["samples"], r["ntok"])]
        deg = np.array([any(f.values()) for f in flags])
        ok = np.array([correct(s, q, fn) for s in r["samples"]])
        out[u] = dict(flags=flags, deg=deg, deg_am=deg & ~ok, ok=ok,
                      uniq=group_count(r["samples"], q_kind(q)) / len(r["samples"]), t=r["t"])
    return out


def boot_ci(vals, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(vals)
    m = [vals[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(cfg, pq, greedy_ok):
    us = sorted(pq)
    deg_q = np.array([pq[u]["deg"].mean() for u in us])
    deg_am_q = np.array([pq[u]["deg_am"].mean() for u in us])
    acc_q = np.array([pq[u]["ok"].mean() for u in us])
    nondeg = np.concatenate([pq[u]["ok"][~pq[u]["deg"]] for u in us])
    crit = {c: float(np.mean([np.mean([f[c] for f in pq[u]["flags"]]) for u in us])) for c in "abc"}
    c_ok = float(np.mean([np.mean([f["c"] and o for f, o in zip(pq[u]["flags"], pq[u]["ok"])]) for u in us]))
    ts = [pq[u]["t"] for u in us if pq[u]["t"] is not None]
    s = dict(cfg=cfg, **CONFIGS[cfg], n_questions=len(us), n_samples=len(us) * N_SAMPLES,
             degenerate=float(deg_q.mean()), degenerate_ci=boot_ci(deg_q),
             degenerate_amended=float(deg_am_q.mean()), degenerate_amended_ci=boot_ci(deg_am_q),
             deg_a=crit["a"], deg_b=crit["b"], deg_c=crit["c"], deg_c_but_correct=c_ok,
             acc_samples=float(acc_q.mean()), acc_samples_ci=boot_ci(acc_q),
             acc_samples_nondeg=float(nondeg.mean()) if len(nondeg) else None,
             acc_greedy=float(np.mean([greedy_ok[u] for u in us])),
             unique_share=float(np.mean([pq[u]["uniq"] for u in us])),
             gen_time_sec=float(sum(ts)) if ts else None)
    return s, acc_q


def choose(summaries, floor):
    thr = floor + DEGEN_MARGIN
    surv = [s for s in summaries if s["degenerate_amended"] <= thr]
    if not surv:
        return None, f"no configuration passed the threshold {thr:.3f} (greedy floor {floor:.3f} + 2 pp)"
    best = max(s["acc_samples"] for s in surv)
    near = [s for s in surv if best - s["acc_samples"] < TIE_PP]
    pick = sorted(near, key=lambda s: (abs(s["temperature"] - 1), -s["acc_samples"]))[0]
    why = (f"threshold {thr:.3f} (greedy floor {floor:.3f} + 2 pp); passed: {[s['cfg'] for s in surv]}; "
           f"best sample accuracy {best:.3f}; within 1 pp of best: {[s['cfg'] for s in near]}; "
           f"picked {pick['cfg']} by |T-1|={abs(pick['temperature'] - 1):.1f}")
    return pick, why


def examples(all_rows, rubq, k=5, seed=0):
    pool = []
    for cfg, rows in all_rows.items():
        for u, r in rows.items():
            for i, s in enumerate(r["samples"]):
                f = degenerate_flags(s, r["ntok"][i] >= MAX_NEW_TOKENS, rubq[u])
                if any(f.values()):
                    pool.append(dict(cfg=cfg, uid=u, question=rubq[u]["question_text"],
                                     gold=gold_strings(rubq[u])[0], crit="".join(c for c in "abc" if f[c]),
                                     text=r["raw"][i] if r["raw"] else s))
    rng = random.Random(seed)
    chosen = []
    for c in "abc":
        cand = [p for p in pool if c in p["crit"] and p not in chosen]
        if cand:
            chosen.append(rng.choice(cand))
    rest = [p for p in pool if p not in chosen]
    rng.shuffle(rest)
    return (chosen + rest)[:k], len(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="lemma", choices=["lemma", "strict"])
    a = ap.parse_args()
    fn = match_lemma if a.acc == "lemma" else match_strict
    if a.acc == "lemma":
        import pymorphy3  # noqa: F401
    rubq = load_rubq()
    uids = subset_ids()
    all_rows = {c: load_cfg_rows(c, uids) for c in CONFIGS}
    common = [u for u in uids if all(u in all_rows[c] for c in CONFIGS)]
    all_rows = {c: {u: all_rows[c][u] for u in common} for c in CONFIGS}
    g_rows = {r["uid"]: r for r in load_jsonl("data/generations.jsonl")}
    greedy_ok = {u: correct(g_rows[u]["greedy"], rubq[u], fn) for u in common}
    gf = [degenerate_flags(g_rows[u]["greedy"], g_rows[u]["greedy_ntok"] >= MAX_NEW_TOKENS, rubq[u]) for u in common]
    floor = dict(version1=float(np.mean([any(f.values()) for f in gf])),
                 amended=float(np.mean([any(f.values()) and not greedy_ok[u] for f, u in zip(gf, common)])),
                 per_criterion={c: float(np.mean([f[c] for f in gf])) for c in "abc"})

    summaries, accs = [], {}
    for c in CONFIGS:
        s, acc_q = summarize(c, per_question(all_rows[c], rubq, fn), greedy_ok)
        summaries.append(s)
        accs[c] = acc_q
    pick, why = choose(summaries, floor["amended"])
    ex, n_pool = examples(all_rows, rubq)
    res = dict(acc_fn=a.acc, greedy_floor=floor, threshold=floor["amended"] + DEGEN_MARGIN, n_questions=len(common),
               summaries=summaries, picked=pick["cfg"] if pick else None, why=why, examples=ex,
               n_degenerate_total=n_pool)
    if pick:
        rng = np.random.default_rng(1)
        n = len(common)
        pd = {}
        for s in summaries:
            if s["cfg"] != pick["cfg"] and s["degenerate_amended"] <= res["threshold"]:
                d = accs[pick["cfg"]] - accs[s["cfg"]]
                m = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
                pd[s["cfg"]] = dict(diff=float(d.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))])
        res["paired_diff_vs_picked"] = pd
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/sampling_params_metrics.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False, indent=1))
    for e in ex:
        print(e)


if __name__ == "__main__":
    main()
