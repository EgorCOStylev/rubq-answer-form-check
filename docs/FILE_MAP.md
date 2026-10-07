# File map

Every file of the repository grouped by stage, in the order the stages were run. Paths are relative to the repository root. For the reading order and the results see the README.

## Analysis plan

| File | Role |
|---|---|
| `src/analysis_plan.md` | Plan for the main-run analysis, fixed 2026-10-05, amendments in section 10 |
| `results/main/deviations.md` | Departures from the plan found while running the checks |

## Stage 0a: answer-form check

| Kind | Files |
|---|---|
| Notebooks | `notebooks/step2_generate.ipynb` (generation, GPU), `notebooks/step34_metrics.ipynb` (metrics, CPU) |
| Code | `src/step1_data.py` (dataset statistics), `src/match.py` (strict and lemma matching, frozen), `src/metrics.py` (D1, D2, D2_sc, decision rule) |
| Data | `data/sample_ids.json` (300 questions), `data/generations.jsonl` (greedy and 10 samples at T = 1.0) |
| Results | `results/metrics.json`, `results/manual_check.csv` (manual check of lemma merges) |

## Sampling parameters

| Kind | Files |
|---|---|
| Notebook | `notebooks/step_sampling_params.ipynb` |
| Code | `src/sp_common.py`, `src/sp_select.py` (question draw), `src/sp_generate.py`, `src/sp_check_eos.py` (stop-condition check), `src/sp_metrics.py` |
| Notes | `src/sampling_params.md` |
| Data | `data/sampling_params/question_ids.json`, `data/sampling_params/generations_sp.jsonl` |
| Results | `results/sampling_params_metrics.json`, `results/sampling_params_eos_check.json` |

## Core signals (on the 300 questions of stage 0a)

| Kind | Files |
|---|---|
| Notebook | `notebooks/step_core_signals.ipynb` |
| Code | `src/signals.py` (self-consistency, tie-aware AURC and AUROC), `src/nli_core.py` (NLI logic, classes, semantic entropy), `src/run_sc.py`, `src/run_nli.py` (fills NLI matrices), `src/analyze_nli.py` (NLI and lemma agreement) |
| Notes | `src/core_signals.md` |
| Results | `results/sc.json`, `results/nli_bare.jsonl`, `results/nli_qa.jsonl`, `results/nli_agreement.json`, `results/review_top50_bare.csv`, `results/review_top50_qa.csv` |

## LM-Polygraph calculator check

| Kind | Files |
|---|---|
| Notebook | `notebooks/step_calculator_check.ipynb` |
| Code | `src/check_calculator.py` |
| Results | `results/calculator_check.json` |

## Main run: generation

| Kind | Files |
|---|---|
| Notebook | `notebooks/step_main_run.ipynb` |
| Code | `src/gen_utils.py`, `src/main_generate.py` (1920 questions, greedy and 10 samples at configuration B), `src/main_params.py` (parameters of the main run), `src/main_check.py` (generation check), `src/run_nli.py` (NLI `qa` matrices) |
| Data | `data/main/generations_main.jsonl` |
| Results | `results/main_generation_check.json`, `results/main/nli_qa.jsonl` |

## Main run, stage A: checks before AUROC

| Kind | Files |
|---|---|
| Notebook | `notebooks/step_main_checks.ipynb` |
| Code | `src/main_checks.py`, `src/analyze_nli.py` |
| Notes | `src/main_checks.md` |
| Tests | `tests/test_main_checks.py` |
| Results | `results/main/checks.json`, `results/main/checks_lemma_free.json`, `results/main/review_top50_qa.csv` |

## Main run, stage B: H3, secondary metrics, cost axis

| Kind | Files |
|---|---|
| Notebook | `notebooks/step_main_results.ipynb` |
| Code | `src/main_signals.py` (signals on the main data), `src/main_results.py` (H3, secondary metrics, cost axis, sensitivity), `src/match_robust.py` (corrected number and date matching, sensitivity only) |
| Notes | `src/main_results.md` |
| Tests | `tests/test_main_signals.py` |
| Results | `results/main/h3.json`, `results/main/secondary.json`, `results/main/cost_axis.json`, `results/main/sensitivity.json`, `results/main/auroc_vs_K.png` |

## Tests

`tests/test_core_signals.py` (self-consistency, NLI logic), `tests/test_gen_utils.py` (generation utilities), `tests/test_main_checks.py` (stage A), `tests/test_main_signals.py` (stage B). Run with `python -m unittest discover -s tests`. They need no GPU.

## Repository files

`README.md`, `LICENSE` (code, MIT), `data/LICENSE` (data derived from RuBQ 2.0, CC BY-SA 4.0), `requirements.txt`, `.gitignore`.

## Naming note

`results/nli_qa.jsonl` holds the NLI matrices of the 300 questions of stage 0a, `results/main/nli_qa.jsonl` those of the 1920 questions of the main run. The same holds for `review_top50_qa.csv`.