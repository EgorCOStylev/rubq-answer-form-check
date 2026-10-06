"""Signals, bootstrap, secondary metrics, cost axis and cheap-signal combination for the main analysis."""
import math
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_params as P
import signals
from nli_core import nli_text, same_nli, se_frequency, unique_texts
from sp_common import MAX_NEW_TOKENS, degenerate_flags

# ---------------------------------------------------------------- per-question sampling signals


def groups_from(pair_same, items):
    """Greedy grouping as in group_sc and greedy_classes: an item joins the first group whose representative it matches."""
    reps, labels = [], []
    for k in items:
        for c, r in enumerate(reps):
            if pair_same[k, r]:
                labels.append(c)
                break
        else:
            reps.append(k)
            labels.append(len(reps) - 1)
    return labels


def question_signals(pl, pn, log_w, items):
    """Self-consistency uncertainty, discrete semantic entropy and probability-weighted semantic entropy."""
    items = list(items)
    if not items:
        return float("nan"), float("nan"), float("nan")
    sizes = Counter(groups_from(pl, items))
    sc = 1.0 - max(sizes.values()) / len(items)
    labels = groups_from(pn, items)
    se = float(se_frequency(labels)) + 0.0
    lw = np.array([log_w[i] for i in items])
    w = np.exp(lw - lw.max())
    mass = np.zeros(max(labels) + 1)
    for lab, x in zip(labels, w):
        mass[lab] += x
    p = mass / mass.sum()
    se_w = float(-(p * np.log(p)).sum()) + 0.0
    return sc, se, se_w


def sampling_signals(T, items_per_question=None, keep=None):
    """items_per_question: list of index lists; keep: boolean (n, 10) mask of samples to use."""
    n = len(T["err"])
    out = np.full((3, n), np.nan)
    for q in range(n):
        if items_per_question is not None:
            items = items_per_question[q]
        elif keep is not None:
            items = np.flatnonzero(keep[q])
        else:
            items = range(P.N_SAMPLES)
        out[:, q] = question_signals(T["PL"][q], T["PN"][q], T["LW"][q], items)
    return dict(sc=out[0], se=out[1], se_w=out[2])


def uncertainty_signals(T, samp):
    return {
        "comparator": -T["sum_lp"],
        "mean_lp": -T["mean_lp"],
        "sc": samp["sc"],
        "se": samp["se"],
        "se_w": samp["se_w"],
        "length": T["ntok"].astype(float),
        "comparator_with_eos": -T["sum_lp_eos"],
    }


# ---------------------------------------------------------------- table from raw data


def build_table(rows, rubq, nli, match_fn, nli_identical_same=False):
    """One row per question. match_fn labels the greedy answer and groups the samples."""
    from match import gold_strings, q_kind
    from metrics import correct

    n = len(rows)
    T = dict(uid=np.array([r["uid"] for r in rows]), kind=[], err=np.zeros(n, bool), sum_lp=np.zeros(n),
             sum_lp_eos=np.zeros(n), mean_lp=np.zeros(n), ntok=np.zeros(n, int), no_eos=np.zeros(n, bool),
             PL=np.zeros((n, P.N_SAMPLES, P.N_SAMPLES), bool), PN=np.zeros((n, P.N_SAMPLES, P.N_SAMPLES), bool),
             LW=np.zeros((n, P.N_SAMPLES)), degen=np.zeros((n, P.N_SAMPLES), bool), empty_samples=0)
    for q, r in enumerate(rows):
        qq = rubq[r["uid"]]
        kind = q_kind(qq)
        S = r["samples"]
        T["kind"].append(kind)
        T["err"][q] = not correct(r["greedy"], qq, match_fn)
        lp = r["greedy_logprobs"]
        T["sum_lp"][q] = sum(lp)
        T["sum_lp_eos"][q] = sum(lp) + (r["greedy_eos_logprob"] if r["greedy_stopped_by_eos"] else 0.0)
        T["mean_lp"][q] = sum(lp) / len(lp)
        T["ntok"][q] = r["greedy_ntok"]
        T["no_eos"][q] = not r["greedy_stopped_by_eos"]
        d = nli[r["uid"]]
        texts, idx = unique_texts(S)
        if texts != d["texts"]:
            raise ValueError(f"uid {r['uid']}: stored NLI texts do not match the samples")
        for i in range(P.N_SAMPLES):
            for j in range(P.N_SAMPLES):
                if i == j:
                    T["PL"][q, i, j] = T["PN"][q, i, j] = True
                    continue
                T["PL"][q, i, j] = bool(match_fn(S[i], S[j], kind))
                T["PN"][q, i, j] = same_nli(d["M"], idx[i], idx[j]) or (nli_identical_same and idx[i] == idx[j])
            T["LW"][q, i] = sum(r["samples_logprobs"][i])
            T["degen"][q, i] = any(degenerate_flags(S[i], r["samples_ntok"][i] >= MAX_NEW_TOKENS, qq).values())
        T["empty_samples"] += sum(nli_text(s) == "" for s in S)
    T["kind"] = np.array(T["kind"])
    return T


def subset_table(T, mask):
    out = {}
    for k, v in T.items():
        out[k] = v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v
    return out


# ---------------------------------------------------------------- bootstrap


def boot_indices(n, B=P.BOOT_B, seed=P.BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, (B, n), dtype=np.int32)


def _auroc_rows(U, E):
    r = rankdata(U, axis=1)
    n1 = E.sum(1).astype(float)
    n0 = E.shape[1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((r * E).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_aurocs(U, err, idx, valid=None, chunk=500):
    """AUROC (ties 0.5) of each signal in U (dict) on every resample in idx; the same resamples for all signals."""
    err = np.asarray(err, bool)
    names = list(U)
    out = {k: np.empty(len(idx)) for k in names}
    if valid is None or valid.all():
        for s in range(0, len(idx), chunk):
            ii = idx[s:s + chunk]
            E = err[ii]
            for k in names:
                out[k][s:s + chunk] = _auroc_rows(np.asarray(U[k], float)[ii], E)
        return out
    for b in range(len(idx)):
        ii = idx[b][valid[idx[b]]]
        for k in names:
            out[k][b] = signals.auroc(np.asarray(U[k], float)[ii], err[ii])
    return out


def ci(x, level):
    a = (100 - level) / 2
    return [float(np.nanpercentile(x, a)), float(np.nanpercentile(x, 100 - a))]


def decide(lo, hi, margin=P.MARGIN):
    if lo >= -margin:
        return "non-inferior"
    if hi < -margin:
        return "inferior"
    return "not established within ±0.03"


def auroc_with_ci(u, err, idx):
    u = np.asarray(u, float)
    valid = np.isfinite(u)
    boots = boot_aurocs({"u": u}, err, idx, valid)["u"]
    return dict(auroc=signals.auroc(u[valid], np.asarray(err)[valid]), ci95=ci(boots, 95), n=int(valid.sum()))


def compare(ua, ub, err, idx):
    """Difference AUROC(a) - AUROC(b), paired bootstrap; questions with a non-finite score in either signal are dropped."""
    ua, ub, err = np.asarray(ua, float), np.asarray(ub, float), np.asarray(err, bool)
    valid = np.isfinite(ua) & np.isfinite(ub)
    boots = boot_aurocs({"a": ua, "b": ub}, err, idx, valid)
    d = boots["a"] - boots["b"]
    a, b = signals.auroc(ua[valid], err[valid]), signals.auroc(ub[valid], err[valid])
    lo95, hi95 = ci(d, 95)
    lo90, hi90 = ci(d, 90)
    return dict(auroc_a=a, auroc_b=b, diff=a - b, ci95=[lo95, hi95], outcome95=decide(lo95, hi95),
                ci90_exploratory=[lo90, hi90], outcome90_exploratory=decide(lo90, hi90),
                n=int(valid.sum()), dropped=int((~valid).sum()))


def h3_block(U, err, idx):
    """AUROC table with intervals, the two H3 comparisons and the sensitivity comparisons."""
    res = dict(n=int(len(err)), error_rate=float(np.mean(err)))
    res["auroc"] = {k: auroc_with_ci(u, err, idx) for k, u in U.items()}
    res["h3"] = {
        "sc_minus_comparator": compare(U["sc"], U["comparator"], err, idx),
        "se_minus_comparator": compare(U["se"], U["comparator"], err, idx),
    }
    res["h3_holds"] = any(c["outcome95"] == "non-inferior" for c in res["h3"].values())
    res["h3_statement"] = "H3 holds" if res["h3_holds"] else "not established within ±0.03"
    res["sensitivity_comparisons"] = {
        "sc_minus_mean_lp": compare(U["sc"], U["mean_lp"], err, idx),
        "se_minus_mean_lp": compare(U["se"], U["mean_lp"], err, idx),
        "se_weighted_minus_comparator": compare(U["se_w"], U["comparator"], err, idx),
        "sc_minus_comparator_with_eos": compare(U["sc"], U["comparator_with_eos"], err, idx),
        "se_minus_comparator_with_eos": compare(U["se"], U["comparator_with_eos"], err, idx),
    }
    return res


# ---------------------------------------------------------------- secondary metrics


def accuracy_at_coverage(u, err, c):
    """Expected accuracy of the k = round(c n) most confident answers; ties at the cut-off are broken at random."""
    u, err = np.asarray(u, float), np.asarray(err, float)
    n = len(u)
    k = max(1, int(round(c * n)))
    order = np.argsort(u, kind="mergesort")
    us, es = u[order], err[order]
    v = us[k - 1]
    below = us < v
    tied = us == v
    expected_err = es[below].sum() + (k - below.sum()) * es[tied].mean()
    return float(1.0 - expected_err / k)


def ece(p, y, bins=P.ECE_BINS):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((which == b).sum() / len(p) * abs(p[which == b].mean() - y[which == b].mean())
                     for b in range(bins) if (which == b).any()))


def cv_calibrated(x, y, seed=P.CV_SEED, folds=P.CV_FOLDS):
    """Out-of-fold Platt scaling of one feature to the probability that y = 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    x = np.asarray(x, float).reshape(len(x), -1)
    p = np.empty(len(x))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(x):
        p[te] = LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return p


def secondary_block(U, T, err):
    from sklearn.metrics import average_precision_score
    err = np.asarray(err, bool)
    out = dict(error_rate=float(err.mean()), aurc_ideal=signals.aurc_ideal(err), aurc_random_order=float(err.mean()),
               signals={})
    grid = np.linspace(1 / P.CURVE_POINTS, 1, P.CURVE_POINTS)
    for k, u in U.items():
        u = np.asarray(u, float)
        v = np.isfinite(u)
        uu, ee = u[v], err[v]
        best, worst = signals.aurc_bounds(uu, ee)
        out["signals"][k] = dict(
            n=int(v.sum()), auprc=float(average_precision_score(ee, uu)), auprc_base_rate=float(ee.mean()),
            aurc=signals.aurc_tie_aware(uu, ee), aurc_best_order=best, aurc_worst_order=worst,
            accuracy_at_coverage={f"{int(c * 100)}": accuracy_at_coverage(uu, ee, c) for c in P.COVERAGES},
            risk_coverage_curve=[[float(c), 1 - accuracy_at_coverage(uu, ee, c)] for c in grid])
    correct = (~err).astype(int)
    probs = {"sequence_probability": T["sum_lp"], "largest_group_share": 1.0 - U["sc"]}
    out["probability_quantities"] = {}
    for k, x in probs.items():
        p = cv_calibrated(x, correct)
        out["probability_quantities"][k] = dict(brier=float(np.mean((p - correct) ** 2)), ece=ece(p, correct),
                                                feature="sum of log-probabilities" if k == "sequence_probability" else "share")
    return out


# ---------------------------------------------------------------- cost axis


def cost_axis(T, err, seed=P.SUBSET_SEED, n_subsets=P.N_SUBSETS):
    n = len(err)
    res = {}
    for K in P.K_VALUES:
        if K == P.N_SAMPLES:
            subs = [None]
        else:
            rng = np.random.default_rng([seed, K])
            subs = [np.sort(rng.random((n, P.N_SAMPLES)).argsort(1)[:, :K], axis=1) for _ in range(n_subsets)]
        vals = {"sc": [], "se": []}
        for sub in subs:
            s = sampling_signals(T, items_per_question=None if sub is None else [list(r) for r in sub])
            for k in vals:
                vals[k].append(signals.auroc(s[k], err))
        res[str(K)] = {k: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n_subsets=len(v))
                       for k, v in vals.items()}
    return res


def cheap_combination_oof(T, err, seed=P.CV_SEED, folds=P.CV_FOLDS):
    """Out-of-fold probability of error from logistic regression on the sum of log-probabilities and answer length."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.column_stack([T["sum_lp"], T["ntok"].astype(float)])
    y = np.asarray(err, int)
    p = np.empty(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        p[te] = make_pipeline(StandardScaler(), LogisticRegression()).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def combination_block(T, U, err, idx):
    combo = cheap_combination_oof(T, err)
    best = max(("sc", "se"), key=lambda k: signals.auroc(U[k], err))
    return dict(
        combination_auroc=auroc_with_ci(combo, err, idx),
        best_sampling_signal_by_point_auroc=best,
        combination_minus_best_sampling=compare(combo, U[best], err, idx),
        best_sampling_minus_combination=compare(U[best], combo, err, idx),
        combination_minus_comparator=compare(combo, U["comparator"], err, idx))
