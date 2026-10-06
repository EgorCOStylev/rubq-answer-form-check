"""Main calculation of src/analysis_plan.md (stage B). Run after the stage A checks and the go-ahead."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_params as P
import main_signals as S
from sp_common import load_jsonl, load_rubq

DATA = "data/main/generations_main.jsonl"
NLI = "results/main/nli_qa.jsonl"
OUT = "results/main"

SIGNAL_LABELS = [
    ("comparator", "Comparator: minus sum of token log-probabilities"),
    ("sc", "Self-consistency (lemma groups)"),
    ("se", "Semantic entropy (NLI qa, cluster frequencies)"),
    ("length", "Baseline: greedy answer length, tokens"),
    ("mean_lp", "Sensitivity: minus mean token log-probability"),
    ("se_w", "Sensitivity: semantic entropy weighted by sample probabilities"),
    ("comparator_with_eos", "Sensitivity: comparator including eos log-probability"),
]


def dump(obj, name, out):
    with open(os.path.join(out, name), "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def sensitivity(T, U, err, rows, rubq, nli, B):
    from match_robust import match_robust
    from match import match_lemma
    res = {}
    keep = ~T["no_eos"]
    idx = S.boot_indices(int(keep.sum()), B)
    Uk = {k: v[keep] for k, v in U.items()}
    res["without_greedy_answers_without_eos"] = dict(removed=int((~keep).sum()), **S.h3_block(Uk, err[keep], idx))

    samp = S.sampling_signals(T, keep=~T["degen"])
    Ud = dict(U, sc=samp["sc"], se=samp["se"], se_w=samp["se_w"])
    idx = S.boot_indices(len(err), B)
    res["without_degenerate_samples"] = dict(
        degenerate_samples=int(T["degen"].sum()), questions_without_any_sample_left=int((~np.isfinite(samp["sc"])).sum()),
        **S.h3_block(Ud, err, idx))

    res["empty_sample"] = dict(
        samples_with_empty_text=int(T["empty_samples"]),
        note="No empty samples in the data, so the separate-group rule of section 8 changes nothing.")

    for name, kw in (("added_robust_matching_number_date_month_names", dict(match_fn=match_robust)),
                     ("added_nli_identical_texts_same_cluster", dict(match_fn=match_lemma, nli_identical_same=True))):
        T2 = S.build_table(rows, rubq, nli, **kw)
        samp2 = S.sampling_signals(T2)
        U2 = S.uncertainty_signals(T2, samp2)
        res[name] = dict(greedy_error_rate=float(T2["err"].mean()),
                         labels_changed_vs_main=int((T2["err"] != err).sum()), **S.h3_block(U2, T2["err"], idx))
    return res


def run(rows, rubq, nli, out=OUT, B=P.BOOT_B, with_sensitivity=True, md_path="src/main_results.md"):
    from match import match_lemma
    os.makedirs(out, exist_ok=True)
    T = S.build_table(rows, rubq, nli, match_lemma)
    err = T["err"]
    samp = S.sampling_signals(T)
    U = S.uncertainty_signals(T, samp)
    idx = S.boot_indices(len(err), B)

    h3 = dict(params=dict(B=B, seed=P.BOOT_SEED, margin=P.MARGIN, level="percentile 95% two-sided; 90% exploratory"),
              semantic_entropy_undefined=int((~np.isfinite(samp["se"])).sum()), **S.h3_block(U, err, idx))
    dump(h3, "h3.json", out)

    dump(S.secondary_block({k: U[k] for k in ("comparator", "sc", "se", "length", "mean_lp", "se_w")}, T, err),
         "secondary.json", out)

    curve = S.cost_axis(T, err)
    cost = dict(params=dict(seed=P.SUBSET_SEED, n_subsets=P.N_SUBSETS, cv_seed=P.CV_SEED, cv_folds=P.CV_FOLDS),
                auroc_by_K=curve, cheap_combination=S.combination_block(T, U, err, idx))
    dump(cost, "cost_axis.json", out)
    plot_cost(curve, h3, cost, os.path.join(out, "auroc_vs_K.png"))

    if with_sensitivity:
        dump(sensitivity(T, U, err, rows, rubq, nli, B), "sensitivity.json", out)
    write_md(out, md_path)


def plot_cost(curve, h3, cost, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ks = [int(k) for k in curve]
    for key, name, color in (("sc", "Self-consistency", "#1f77b4"), ("se", "Semantic entropy", "#d95f02")):
        m = np.array([curve[str(k)][key]["mean"] for k in Ks])
        lo = np.array([curve[str(k)][key]["min"] for k in Ks])
        hi = np.array([curve[str(k)][key]["max"] for k in Ks])
        ax.plot(Ks, m, marker="o", color=color, label=name)
        ax.fill_between(Ks, lo, hi, color=color, alpha=0.15)
    ax.axhline(h3["auroc"]["comparator"]["auroc"], color="#444", ls="--", label="Comparator (sum of log-probs)")
    ax.axhline(cost["cheap_combination"]["combination_auroc"]["auroc"], color="#2a9d8f", ls=":",
               label="Log-probs + length (cross-validated)")
    ax.set_xlabel("Number of samples K")
    ax.set_ylabel("AUROC (error of the greedy answer)")
    ax.set_xticks(Ks)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt(x, d=3):
    return f"{x:.{d}f}"


def interval(c):
    return f"[{fmt(c[0])}, {fmt(c[1])}]"


def write_md(out, md_path="src/main_results.md"):
    h3 = json.load(open(os.path.join(out, "h3.json")))
    cost = json.load(open(os.path.join(out, "cost_axis.json")))
    sec = json.load(open(os.path.join(out, "secondary.json")))
    sens_path = os.path.join(out, "sensitivity.json")
    sens = json.load(open(sens_path)) if os.path.exists(sens_path) else None
    L = ["# Main results (stage B)", "",
         "Computed by `src/main_results.py` strictly under `src/analysis_plan.md`. Positive class: error of the greedy answer. "
         f"n = {h3['n']} questions, error rate {fmt(h3['error_rate'])}. Paired bootstrap over questions, {h3['params']['B']} resamples, "
         "percentile intervals, one set of resamples for all signals. Differences below about 0.02-0.035 are not read as a ranking of signals.", "",
         "## AUROC of the signals", "", "| Signal | AUROC | 95% interval | n |", "|---|---|---|---|"]
    for k, name in SIGNAL_LABELS:
        a = h3["auroc"][k]
        L.append(f"| {name} | {fmt(a['auroc'])} | {interval(a['ci95'])} | {a['n']} |")
    L += ["", "## H3: sampling signal minus comparator (margin 0.03)", "",
          "| Comparison | Difference | 95% interval | Outcome | 90% interval (exploratory) | Outcome at 90% (exploratory) |",
          "|---|---|---|---|---|---|"]
    for k, name in (("sc_minus_comparator", "Self-consistency minus comparator"),
                    ("se_minus_comparator", "Semantic entropy minus comparator")):
        c = h3["h3"][k]
        L.append(f"| {name} | {fmt(c['diff'])} | {interval(c['ci95'])} | {c['outcome95']} | "
                 f"{interval(c['ci90_exploratory'])} | {c['outcome90_exploratory']} |")
    L += ["", f"Overall: {h3['h3_statement']}. Questions with undefined semantic entropy: {h3['semantic_entropy_undefined']}.", "",
          "## Cost axis", "", "AUROC by number of samples K (mean over 20 random subsets for K < 10).", "",
          "| K | Self-consistency | Semantic entropy |", "|---|---|---|"]
    for K, v in cost["auroc_by_K"].items():
        L.append(f"| {K} | {fmt(v['sc']['mean'])} | {fmt(v['se']['mean'])} |")
    cc = cost["cheap_combination"]
    c1, c2 = cc["combination_minus_best_sampling"], cc["best_sampling_minus_combination"]
    L += ["", f"Cheap combination (sum of log-probabilities and length, 5-fold cross-validation): AUROC {fmt(cc['combination_auroc']['auroc'])} "
          f"{interval(cc['combination_auroc']['ci95'])}. Best sampling signal at K = 10 by point AUROC: {cc['best_sampling_signal_by_point_auroc']}.", "",
          "| Comparison | Difference | 95% interval | Outcome |", "|---|---|---|---|",
          f"| Combination minus best sampling signal | {fmt(c1['diff'])} | {interval(c1['ci95'])} | {c1['outcome95']} |",
          f"| Best sampling signal minus combination | {fmt(c2['diff'])} | {interval(c2['ci95'])} | {c2['outcome95']} |", "",
          "![AUROC against K](auroc_vs_K.png)", "", "## Secondary metrics", "",
          "| Signal | AUPRC (base rate " + fmt(sec['error_rate']) + ") | AURC | AURC best / worst order | Acc@20% | Acc@50% | Acc@80% |",
          "|---|---|---|---|---|---|---|"]
    for k, v in sec["signals"].items():
        a = v["accuracy_at_coverage"]
        L.append(f"| {k} | {fmt(v['auprc'])} | {fmt(v['aurc'])} | {fmt(v['aurc_best_order'])} / {fmt(v['aurc_worst_order'])} | "
                 f"{fmt(a['20'])} | {fmt(a['50'])} | {fmt(a['80'])} |")
    L += ["", f"Ideal AURC {fmt(sec['aurc_ideal'])}, random order {fmt(sec['aurc_random_order'])}.", "",
          "| Probability quantity (calibrated by 5-fold cross-validation) | Brier | ECE |", "|---|---|---|"]
    for k, v in sec["probability_quantities"].items():
        L.append(f"| {k} | {fmt(v['brier'])} | {fmt(v['ece'])} |")
    if sens:
        L += ["", "## Sensitivity (section 8 and added rows)", "",
              "| Variant | n | Self-consistency minus comparator | Semantic entropy minus comparator |", "|---|---|---|---|"]
        for k, v in sens.items():
            if "h3" not in v:
                continue
            a, b = v["h3"]["sc_minus_comparator"], v["h3"]["se_minus_comparator"]
            L.append(f"| {k} | {v['n']} | {fmt(a['diff'])} {interval(a['ci95'])} {a['outcome95']} | "
                     f"{fmt(b['diff'])} {interval(b['ci95'])} {b['outcome95']} |")
        L += ["", f"Empty samples in the data: {sens['empty_sample']['samples_with_empty_text']}."]
    dev = os.path.join(out, "deviations.md")
    L += ["", "## Deviations and additions", ""]
    L.append(open(dev).read().split("\n", 1)[1].strip() if os.path.exists(dev) else "None recorded.")
    with open(md_path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    rows = load_jsonl(DATA)
    rubq = load_rubq()
    nli = {r["uid"]: r for r in load_jsonl(NLI)}
    run(rows, rubq, nli)


if __name__ == "__main__":
    main()
