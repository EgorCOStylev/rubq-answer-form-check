# Main results (stage B)

Computed by `src/main_results.py` strictly under `src/analysis_plan.md`. Positive class: error of the greedy answer. n = 1920 questions, error rate 0.670. Paired bootstrap over questions, 10000 resamples, percentile intervals, one set of resamples for all signals. Differences below about 0.02-0.035 are not read as a ranking of signals.

## AUROC of the signals

| Signal | AUROC | 95% interval | n |
|---|---|---|---|
| Comparator: minus sum of token log-probabilities | 0.825 | [0.806, 0.843] | 1920 |
| Self-consistency (lemma groups) | 0.788 | [0.768, 0.808] | 1920 |
| Semantic entropy (NLI qa, cluster frequencies) | 0.788 | [0.769, 0.807] | 1920 |
| Baseline: greedy answer length, tokens | 0.623 | [0.598, 0.648] | 1920 |
| Sensitivity: minus mean token log-probability | 0.810 | [0.790, 0.830] | 1920 |
| Sensitivity: semantic entropy weighted by sample probabilities | 0.781 | [0.761, 0.800] | 1920 |
| Sensitivity: comparator including eos log-probability | 0.825 | [0.806, 0.843] | 1920 |

## H3: sampling signal minus comparator (margin 0.03)

| Comparison | Difference | 95% interval | Outcome | 90% interval (exploratory) | Outcome at 90% (exploratory) |
|---|---|---|---|---|---|
| Self-consistency minus comparator | -0.037 | [-0.048, -0.026] | not established within ±0.03 | [-0.046, -0.028] | not established within ±0.03 |
| Semantic entropy minus comparator | -0.037 | [-0.050, -0.023] | not established within ±0.03 | [-0.047, -0.025] | not established within ±0.03 |

Overall: not established within ±0.03. Questions with undefined semantic entropy: 0.

## Cost axis

AUROC by number of samples K (mean over 20 random subsets for K < 10).

| K | Self-consistency | Semantic entropy |
|---|---|---|
| 1 | 0.500 | 0.500 |
| 2 | 0.699 | 0.694 |
| 3 | 0.743 | 0.738 |
| 4 | 0.757 | 0.755 |
| 5 | 0.769 | 0.766 |
| 6 | 0.776 | 0.774 |
| 7 | 0.781 | 0.780 |
| 8 | 0.783 | 0.783 |
| 9 | 0.786 | 0.786 |
| 10 | 0.788 | 0.788 |

Cheap combination (sum of log-probabilities and length, 5-fold cross-validation): AUROC 0.823 [0.804, 0.842]. Best sampling signal at K = 10 by point AUROC: se.

| Comparison | Difference | 95% interval | Outcome |
|---|---|---|---|
| Combination minus best sampling signal | 0.035 | [0.021, 0.048] | non-inferior |
| Best sampling signal minus combination | -0.035 | [-0.048, -0.021] | not established within ±0.03 |

![AUROC against K](auroc_vs_K.png)

## Secondary metrics

| Signal | AUPRC (base rate 0.670) | AURC | AURC best / worst order | Acc@20% | Acc@50% | Acc@80% |
|---|---|---|---|---|---|---|
| comparator | 0.908 | 0.431 | 0.429 / 0.435 | 0.729 | 0.543 | 0.408 |
| sc | 0.869 | 0.487 | 0.381 / 0.617 | 0.617 | 0.530 | 0.403 |
| se | 0.871 | 0.492 | 0.372 / 0.634 | 0.596 | 0.540 | 0.404 |
| length | 0.771 | 0.606 | 0.535 / 0.673 | 0.386 | 0.417 | 0.379 |
| mean_lp | 0.889 | 0.434 | 0.432 / 0.439 | 0.737 | 0.526 | 0.396 |
| se_w | 0.863 | 0.494 | 0.377 / 0.634 | 0.596 | 0.532 | 0.401 |

Ideal AURC 0.304, random order 0.670.

| Probability quantity (calibrated by 5-fold cross-validation) | Brier | ECE |
|---|---|---|
| sequence_probability | 0.163 | 0.048 |
| largest_group_share | 0.172 | 0.033 |

## Sensitivity (section 8 and added rows)

| Variant | n | Self-consistency minus comparator | Semantic entropy minus comparator |
|---|---|---|---|
| without_greedy_answers_without_eos | 1878 | -0.036 [-0.047, -0.026] not established within ±0.03 | -0.036 [-0.049, -0.022] not established within ±0.03 |
| without_degenerate_samples | 1920 | -0.042 [-0.054, -0.029] not established within ±0.03 | -0.046 [-0.060, -0.032] inferior |
| added_robust_matching_number_date_month_names | 1920 | -0.040 [-0.051, -0.029] not established within ±0.03 | -0.035 [-0.049, -0.022] not established within ±0.03 |
| added_nli_identical_texts_same_cluster | 1920 | -0.037 [-0.048, -0.026] not established within ±0.03 | -0.037 [-0.050, -0.024] not established within ±0.03 |

Empty samples in the data: 0.

## Deviations and additions

Recorded as found. Nothing below was applied silently.

1. Plan version. The checks were prepared from the project copy of the plan (dated 2026-10-05) and then compared with the repository version (dated 2026-10-06, Amendment of 2026-10-06 removes max-softmax probability). The amendment concerns stage B only (baselines, cheap-signal combination). Sections 8 and 9, which stage A uses, are identical in both versions.
2. Empty sample. Section 8 refers to an empty sample for question 309. The main-run data contain no empty sample: 0 of 19200 samples have zero tokens or an empty first line. Question 309 has the samples «Санкт-Петербург» and «Петроград». The sensitivity row for an empty sample is therefore vacuous and will be reported as such.
3. Correctness-conditioned quantities in check 2. `analyze_nli.py` also reports the mean semantic entropy for correct and for wrong greedy answers. This is a signal compared against correctness, so it is omitted from the pre-AUROC checks (`--no-correctness`). The logic of the pair comparison is unchanged.
4. `analyze_nli.py` now takes `--data`, `--nli-dir`, `--out-dir`, `--variants`. Defaults reproduce the stage 0a behaviour; all tests pass.
5. Environment. In the session that prepared the checks `pymorphy3` could not be installed (pypi.org is blocked for the sandbox). Checks 2 and 3 were computed on Kaggle (`notebooks/step_main_checks.ipynb`); check 1 is also in `results/main/checks_lemma_free.json` (computed without the lemmatizer, identical values).

## Stage B: interpretations and additions

These are fixed in the code before any AUROC was computed on the main-run data. None replaces a planned figure.

6. Probability-weighted semantic entropy (sensitivity): cluster mass is the normalised sum of sample sequence probabilities, where a sample probability is exp of the sum of its token log-probabilities (eos not included); computed through log-sum-exp.
7. Comparator and eos. The sum of token log-probabilities of the greedy answer excludes the eos token. A variant that adds the eos log-probability (when the answer stopped by eos) is reported as a sensitivity row.
8. Orientation of the baseline: a longer greedy answer (more tokens) is read as more uncertain. The orientation is fixed here and not chosen by AUROC.
9. Cost axis, combination against the best sampling signal. The plan says "the same criterion (margin 0.03)" without a direction. Both directions are reported: combination minus best sampling signal (can the cheap combination replace sampling) and best sampling signal minus combination. The best sampling signal is the one with the higher point AUROC at K = 10, so the selection uses the data. The combination uses out-of-fold predictions that are not refitted inside the bootstrap.
10. Brier and ECE are computed for two quantities that are probabilities: the sequence probability of the greedy answer (feature: sum of log-probabilities) and the share of the largest sample group (feature: the share), each calibrated by 5-fold cross-validation with logistic regression; ECE uses 10 equal-width bins.
11. Accuracy at coverage breaks ties at the cut-off in expectation over random order. The risk-coverage curve is given at 20 coverage points.
12. Added sensitivity rows decided after the stage A review, before any AUROC: (a) matching with month names read as numbers and equal cleaned texts matched for number and date questions (`src/match_robust.py`; labels and sample groups both change; `match.py` stays frozen); (b) identical sample texts always in the same NLI cluster. Plan section 3 allows the first as a robustness analysis; the second is not in the plan.
13. Questions for which a sampling signal is undefined after removing degenerate samples are dropped only from comparisons that involve it, with the count reported.
