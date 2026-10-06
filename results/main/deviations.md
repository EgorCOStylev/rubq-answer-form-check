# Deviations from src/analysis_plan.md

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
