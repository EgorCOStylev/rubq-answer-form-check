"""Stability of the H3 differences to the choice of samples (amendment of 2026-10-09 in src/analysis_plan.md)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_params as P
import main_signals as S
import signals
from main_results import DATA, NLI, OUT
from sp_common import load_jsonl, load_rubq

K = 9
# Half of the bootstrap standard error of each H3 difference, as fixed in the amendment.
THRESHOLD = {"sc": 0.0028, "se": 0.0034}


def subsets_k(n, k=K, seed=P.SUBSET_SEED, n_subsets=P.N_SUBSETS):
    """The same generator and draw order as main_signals.cost_axis, so the subsets are those of the cost axis."""
    rng = np.random.default_rng([seed, k])
    return [np.sort(rng.random((n, P.N_SAMPLES)).argsort(1)[:, :k], axis=1) for _ in range(n_subsets)]


def run(rows, rubq, nli, out=OUT):
    from match import match_lemma
    T = S.build_table(rows, rubq, nli, match_lemma)
    err = T["err"]
    comparator = signals.auroc(-T["sum_lp"], err)
    aurocs = {"sc": [], "se": []}
    for sub in subsets_k(len(err)):
        s = S.sampling_signals(T, items_per_question=[list(r) for r in sub])
        for k in aurocs:
            aurocs[k].append(signals.auroc(s[k], err))

    cost = json.load(open(os.path.join(out, "cost_axis.json")))["auroc_by_K"][str(K)]
    res = dict(K=K, n_subsets=P.N_SUBSETS, subset_seed=P.SUBSET_SEED, n=int(len(err)), comparator_auroc=comparator,
               margin=P.MARGIN, sd="sample standard deviation (ddof = 1) over subsets", signals={})
    for k, a in aurocs.items():
        a = np.array(a)
        d = a - comparator
        sd = float(d.std(ddof=1))
        res["signals"][k] = dict(
            auroc_per_subset=a.tolist(), difference_per_subset=d.tolist(),
            mean=float(d.mean()), sd=sd, min=float(d.min()), max=float(d.max()),
            subsets_above_minus_margin=int((d > -P.MARGIN).sum()),
            threshold=THRESHOLD[k], below_threshold=sd < THRESHOLD[k],
            reproduces_cost_axis=bool(np.isclose(a.mean(), cost[k]["mean"]) and np.isclose(a.min(), cost[k]["min"])
                                      and np.isclose(a.max(), cost[k]["max"])))
    small = all(v["below_threshold"] for v in res["signals"].values())
    res["decision"] = ("sampling randomness small: reported as a limitation, no further runs" if small else
                       "second set of 10 samples with a different seed to be considered in a separate amendment")
    with open(os.path.join(out, "stability_k9.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def main():
    res = run(load_jsonl(DATA), load_rubq(), {r["uid"]: r for r in load_jsonl(NLI)})
    for k, v in res["signals"].items():
        print(f"{k}: mean {v['mean']:.4f}, sd {v['sd']:.4f} (threshold {v['threshold']}), "
              f"range [{v['min']:.4f}, {v['max']:.4f}], above -{P.MARGIN}: {v['subsets_above_minus_margin']}, "
              f"reproduces cost axis: {v['reproduces_cost_axis']}")
    print(res["decision"])


if __name__ == "__main__":
    main()
