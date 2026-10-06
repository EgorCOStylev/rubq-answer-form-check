# Deviations from src/analysis_plan.md

Recorded as found. Nothing below was applied silently.

1. Plan version. The checks were prepared from the project copy of the plan (dated 2026-10-05) and then compared with the repository version (dated 2026-10-06, Amendment of 2026-10-06 removes max-softmax probability). The amendment concerns stage B only (baselines, cheap-signal combination). Sections 8 and 9, which stage A uses, are identical in both versions.
2. Empty sample. Section 8 refers to an empty sample for question 309. The main-run data contain no empty sample: 0 of 19200 samples have zero tokens or an empty first line. Question 309 has the samples «Санкт-Петербург» and «Петроград». The sensitivity row for an empty sample is therefore vacuous and will be reported as such.
3. Correctness-conditioned quantities in check 2. `analyze_nli.py` also reports the mean semantic entropy for correct and for wrong greedy answers. This is a signal compared against correctness, so it is omitted from the pre-AUROC checks (`--no-correctness`). The logic of the pair comparison is unchanged.
4. `analyze_nli.py` now takes `--data`, `--nli-dir`, `--out-dir`, `--variants`. Defaults reproduce the stage 0a behaviour; all tests pass.
5. Environment. In the session that prepared the checks `pymorphy3` could not be installed (pypi.org is blocked for the sandbox). Checks 2 and 3 were computed on Kaggle (`notebooks/step_main_checks.ipynb`); check 1 is also in `results/main/checks_lemma_free.json` (computed without the lemmatizer, identical values).
