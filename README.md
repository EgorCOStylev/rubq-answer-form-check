# Does answer-form variation matter in Russian short-answer QA? A pre-registered check on RuBQ 2.0

**TL;DR — negative result.** With a short-answer prompt, Qwen2.5-7B-Instruct (4-bit) almost never answers in an inflected or prepositional form. Lemmatized matching changes the correctness verdict for **2.2%** of answers and the self-consistency score for **1.7%** of questions. Both are below the 3% threshold fixed before the experiment, so by the pre-registered rule the premise is rejected.

> Status on 7 October 2026: the main run and its analysis are finished. H3 is not established within ±0.03: the log-probability comparator reaches AUROC 0.825, self-consistency and semantic entropy 0.788 each (see "How to read this repository").

| Stage | Outcome | Files |
|---|---|---|
| Answer-form check (this document, up to "Deviations") | Premise rejected: D1 = 2.2%, D2_sc = 1.7% | `src/match.py`, `src/metrics.py`, `results/metrics.json` |
| Sampling parameters | No configuration passed the pre-registered rule, B chosen outside it | `src/sp_*.py`, `src/sampling_params.md`, `results/sampling_params_*.json` |
| Core signals | Self-consistency AUROC 0.823 on 300 questions, NLI input `qa` | `src/signals.py`, `src/nli_core.py`, `src/core_signals.md`, `results/sc.json`, `results/nli_agreement.json` |
| LM-Polygraph calculator check | Calculator rejected for sample generation | `src/check_calculator.py`, `results/calculator_check.json` |
| Main run | 1920 questions, greedy plus 10 samples at B, reproduces earlier stages | `src/main_generate.py`, `src/main_check.py`, `data/main/`, `results/main/` |
| Checks before AUROC (stage A) | Semantic entropy finite on all 1920 questions, lemma and NLI diverge on 7.5% of pairs | `src/analyze_nli.py`, `src/main_checks.md`, `results/main/` |
| Main analysis (stage B) | H3 not established within ±0.03: AUROC 0.825 against 0.788 and 0.788 | `src/main_results.py`, `src/main_results.md`, `results/main/h3.json` |

## How to read this repository

The analysis of the main run was fixed in a plan ([src/analysis_plan.md](src/analysis_plan.md)) committed on 2026-10-05, before any AUROC or AURC was computed on the main-run data. Earlier stages (1 to 4 below) were run before the plan, and the plan discloses which of their results were known ([section 2](src/analysis_plan.md)). Amendments dated 2026-10-06 and 2026-10-07 are recorded in section 10 of the plan; departures found while running the checks are listed in [deviations](results/main/deviations.md). Stage A checks were committed (5bdbf71, 63e17eb) before the stage B analysis (ac016d9). State of the repository at the time of the results: tag [`main-results-v1`](../../releases/tag/main-results-v1).

1. **Does answer-form variation distort exact-match labels?** Little: strict and lemma labels differ on 2.2% of 300 questions [0.0; 5.6]. → [metrics](results/metrics.json)
2. **Can the sampling signals be computed with the LM-Polygraph calculator?** Not as is: it forces a minimum of two tokens and zeroes one-token answers, so the samples of the main run are generated with our own code. → [check](results/calculator_check.json)
3. **Which sampling configuration?** Configuration B (T = 0.7, top_k = 0, top_p = 0.95). No candidate met the pre-set threshold, B was chosen outside the rule. → [notes](src/sampling_params.md)
4. **Do self-consistency and NLI grouping agree with lemma matching?** On stage 0a the `qa` input diverges from lemma matching on 8.5% of sample pairs, `bare` on 11.3%. → [notes](src/core_signals.md)
5. **Main run.** 1920 questions, one greedy answer and 10 samples each. → [generation check](results/main_generation_check.json)
6. **Checks before AUROC.** Semantic entropy is finite on all questions; lemma and NLI diverge on 7.5% of pairs. → [notes](src/main_checks.md)
7. **H3: is sampling at least as good as log-probability?** Not established within ±0.03. AUROC 0.825 for the log-probability comparator against 0.788 for self-consistency and 0.788 for semantic entropy; paired differences −0.037 [−0.048; −0.026] and −0.037 [−0.050; −0.023] against a margin of 0.03. → [notes](src/main_results.md)

Notebooks, code and outputs for each step: [docs/FILE_MAP.md](docs/FILE_MAP.md).

## Motivation

A planned master's thesis compares LLM confidence signals as hallucination detectors on Russian factoid QA with short answers (RuBQ 2.0). Its novelty rested on one **premise**: models often give a *correct* answer in a surface form that differs from the reference, e.g. «в Москве» vs «Москва», or «Толстого» vs «Толстой». If that were true:

- **H1.** Exact-match labels would mark correct answers as wrong and distort the AUROC of probability-based confidence signals.
- **H2.** Self-consistency would be underestimated, because equivalent samples would not be grouped together.

Before building on this premise, we measured whether it holds. The metrics, thresholds and decision rule were written down **before** any generation.

## Setup

| | |
|---|---|
| Data | RuBQ 2.0 test, 300 of the 1920 answerable questions: sorted by `uid`, then `random.Random(42).sample`. By answer type: entity 213, number 54, date 23, string 10 |
| Model | `Qwen/Qwen2.5-7B-Instruct`, bitsandbytes NF4 4-bit, Kaggle T4, transformers 5.0.0 |
| Prompt | One Russian system instruction («Ответь кратко: только сущность, число или дата») plus 4 invented few-shot pairs, with answers in the nominative case |
| Greedy | `max_new_tokens=16`, `repetition_penalty=1.0` |
| Samples | 10 per question, `T=1.0`, `top_p=1.0`, `top_k=0`, seed = 42·10⁵ + uid. Qwen's default `generation_config` (T=0.7, top_p=0.8, top_k=20, rep. 1.05) is explicitly overridden |

**Matching** (`src/match.py`, frozen before the metrics were computed). Both functions apply the same preprocessing: keep the first line only, strip accents, lowercase, replace ё with е, turn punctuation into spaces.

- `match_strict`: exact string equality with the reference label or any alias (`wd_names.ru/en`, `wp_names`).
- `match_lemma`: equality of the lemma multisets from pymorphy3, after removing a fixed list of 31 prepositions and conjunctions.
- Numbers and dates use one shared rule in both functions. A number matches if the first number in the answer equals the reference. A date matches if the year is equal and the other components do not contradict the reference.

**Metrics.** CIs are 95% bootstrap intervals, 1000 resamples over questions.

- **D1**: share of greedy answers whose verdict differs between strict and lemma matching, among answers correct under at least one of them.
- **D2**: share of sample pairs that match under lemma matching but not under strict matching, among all lemma-matching pairs.
- **D2_sc**: share of questions whose self-consistency (the size of the largest group of samples, divided by 10) changes between strict and lemma grouping.

**Decision rule (pre-registered).**

| Result | Decision |
|---|---|
| D1 ≥ 10% or D2_sc ≥ 10% | Premise strong |
| both in 3–10% | Proceed, focusing on H2 |
| D1 < 3% and D2_sc < 3% | Mechanism does not work |
| Acc_lemma < 15% | Switch to a larger model, no decision |

If a CI crosses a threshold, this is recorded and the more cautious decision is taken.

## Results

| Metric | Value | 95% CI | Counts |
|---|---|---|---|
| Acc_strict | 29.7% | [24.3, 35.3] | 89 / 300 |
| Acc_lemma | 30.3% | [25.0, 36.0] | 91 / 300 |
| **D1** | **2.2%** | [0.0, 5.6] | 2 / 91 |
| D2 | 0.96% | [0.14, 2.05] | 52 / 5420 pairs |
| **D2_sc** | **1.7%** | [0.3, 3.3] | 5 / 300 questions |
| mean \|ΔSC\| | 0.003 | [0.001, 0.006] | |

**Decision: mechanism does not work.** The upper CI bounds of D1 (5.6%) and D2_sc (3.3%) cross the 3% threshold. The cautious decision, taken on the lower bounds, is the same.

**Why.** The model almost never uses the forms the premise relies on. Only 2 of 300 greedy answers and 21 of 3000 samples start with a preposition. The inflected forms in `wp_names` rescue just 1 of the 89 strict-correct answers, so reference aliases do not explain the low D1.

**Examples of divergences.**

| Question | Reference / other sample | Answer | strict | lemma |
|---|---|---|---|---|
| Кто стал чемпионом мира по футболу… в 1998 году? | alias «Франции» | Франция | ✗ | ✓ |
| Какого пола Фейс? | мужской (alias) | Мужского | ✗ | ✓ |
| Притоком какой реки является Золотая Липа? | Днепра | Днепр | ✗ | ✓ |
| В каком городе родился Валентин Пикуль? | Санкт-Петербурге | Санкт-Петербург | ✗ | ✓ |
| Кого рисуют художники-анималисты? | Животные | Животных | ✗ | ✓ |

**Lemmatization quality.** All 14 candidates were checked by hand: 2 from D1 plus 12 D2 sample pairs (8 distinct once order is ignored). The protocol asked for 50, but only 14 exist. False merges: **0 / 14** (`results/manual_check.csv`). One borderline pair, «Афина» / «Афине» (city of the Acropolis), is a form error by the model itself, not a merge of different entities.

## Deviations, caveats, scope

- **Deviation from the protocol.** The protocol put the number/date rule into lemma matching only. We applied it to both functions. Otherwise every correct date, for example «1860» vs the reference `1860-07-02T00:00:00Z`, would count toward D1, and D1 would measure date formatting rather than morphology. This makes D1 more conservative.
- **Few-shot answers are in the nominative case.** This nudges the model toward canonical forms. It was recorded before measurement as a downward bias. The conclusion is therefore scoped to **short-answer prompting with canonical-form examples**. It says nothing about free-form generation.
- **Noisy samples.** 3.6% of samples contain no Cyrillic and no digits: mixed scripts, code, leaked chat turns. In a 5-question smoke test, `top_p=0.95` and float32 compute left the samples unchanged, so this is a property of the model's distribution, not a numerical bug. It slightly deflates D2 / D2_sc.
- **Bug fix.** An earlier version of `decision()` failed to flag the CI/threshold crossing. It is fixed, and `results/metrics.json` has been recomputed; the statistics themselves did not change.
- **Lemmatizer error.** pymorphy3 `parse[0]` produces one visible error on the control pairs: «Восточной» → «восточноить». It did not cause any merges.
- **Limited setup.** One model, one prompt, 300 questions.

## Later stage: sampling parameters

The noisy samples above motivated a pre-registered choice of sampling parameters for the main run (details in `src/sampling_params.md`). Four configurations were compared on 100 questions drawn with seed 20261004 from the 300 of the check (`data/sampling_params/question_ids.json`), 10 samples each, same prompt and `max_new_tokens=16`. Configuration A is taken from `data/generations.jsonl`.

| | T | top_k | top_p |
|---|---|---|---|
| A | 1.0 | 0 | 1.0 |
| B | 0.7 | 0 | 0.95 |
| C | 1.0 | 50 | 0.95 |
| D | 0.8 | 50 | 0.95 |

**Selection rule (fixed before generation).** A sample is degenerate if it is wrong under lemma matching and at least one of the following holds: it reached `max_new_tokens` without `<|im_end|>`; it contains CJK characters; the reference and its aliases contain Cyrillic while the share of Cyrillic among the letters of the answer is below 50%. The same definition applied to greedy answers gives the floor, 5.0%. A configuration is dropped if its degenerate share exceeds the floor by more than 2 pp (threshold 7.0%). Among the rest the highest sample accuracy wins; within 1 pp the smaller |T - 1| wins. AUROC is not used. If nothing passes, the run stops and the grid is not extended.

| Config | Degenerate share, 95% CI | Sample accuracy | Share of unique samples |
|---|---|---|---|
| A | 17.9% [13.8, 22.5] | 27.2% | 50.1% |
| B | 7.4% [4.2, 11.3] | 29.2% | 35.5% |
| C | 10.7% [7.3, 14.6] | 28.1% | 45.9% |
| D | 7.9% [4.7, 11.8] | 29.0% | 39.3% |

Greedy accuracy on the same 100 questions is 31.0%. **No configuration passed the threshold, so the rule selects nothing** (`picked: null` in `results/sampling_params_metrics.json`). B exceeds the threshold by 0.4 pp, within the confidence interval. B was chosen for the main run by the owner outside the rule, and the report records it as such.

**Stop-condition check** (`src/sp_check_eos.py`, 20 questions from the LM-Polygraph check, 1000 samples per condition, T = 1.0, top_k = 0, top_p = 1.0). Samples that continue the chat after a junk token (`<|im_start|>` or `<|endoftext|>` before the end of the answer) occur without `min_new_tokens`: 0.8% with the earlier LM-Polygraph prompt and 3.3% with this prompt. `min_new_tokens=2`, which LM-Polygraph 0.7.0 sets in its sampling calculator, raises these to 1.3% and 5.0% and removes all one-token answers (5.4-5.6% without it). The main mechanism is a junk token drawn from the full vocabulary, followed by `\n<|im_start|>`.

## Later stage: core signals

Details in `src/core_signals.md`, computed on the 300 questions of the check (T = 1.0).

**Self-consistency** is one minus the share of the largest group of samples under the lemma match. It takes 10 distinct values on 10 samples, so the AURC uses the exact expectation over random order inside tied scores (tie-aware). Error rate of the greedy answer 69.7% (209 / 300). AUROC against greedy error 0.823 [0.774, 0.867]. AURC 0.490 [0.415, 0.567]; the best and worst order inside ties give 0.422 and 0.578, the ideal ranking 0.336 and the random one 0.697.

**NLI for semantic entropy.** Model `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. Two inputs were compared by their disagreement with the lemma grouping on the 13 500 sample pairs: `bare` (the answers alone) and `qa` (question plus answer, as in Kuhn et al., 2023).

| Input | Disagreeing pairs | Questions with a disagreement |
|---|---|---|
| `bare` | 1519 (11.3%, CI [9.3, 13.4]) | 167 |
| `qa` | 1154 (8.5%, CI [6.7, 10.6]) | 138 |

The paired difference `qa` minus `bare` is -2.7 pp [-3.7, -1.9], so by the pre-registered rule the main run uses `qa`. This choice is by agreement with the lemma grouping and does not show that `qa` is better for semantic entropy: many disagreements are correct merges by NLI that the lemma match cannot see («Таллинн» and «Таллин», «IX век» and «IX столетие»). Pair types and manual-review files are in `results/nli_agreement.json` and `results/review_top50_*.csv`. Semantic entropy is finite on 300 of 300 questions.

**Limitations of `match.py`** found by direct counting (the protocol is frozen, so nothing was changed). Two identical word answers to a number question («Март» and «Март») do not match because they contain no digits: 19 of 1474 identical pairs, 3 questions. Different full dates of one year without `T` can match by year alone: 46 of 1035 date pairs, 3 questions. Identical dates without a year do not match: 39 of 697 pairs, 2 questions. At most 8 of 300 questions are affected. Fixing this would require recomputing the check above.

## Later stage: LM-Polygraph calculator check

Can `SamplingGenerationCalculator` from LM-Polygraph 0.7.0 generate the samples of the main run? `src/check_calculator.py` compares it with `model.generate` without `min_new_tokens` at configuration B, on 20 questions with 3 repeats (600 samples per side).

| | One-token answers | Samples with `<\|im_start\|>` or `<\|endoftext\|>` | Reached 16 tokens | Without eos |
|---|---|---|---|---|
| `model.generate` | 30 (5.0%) | not counted | not counted | 14 (2.3%) |
| Calculator | 0 | 24 (4.0%) | 17 (2.8%) | not counted |

The calculator's minimum sample length is 2 tokens. It forces answers like «1» to continue, for example into `1<|im_start|>\n<|endoftext|>`. The main run therefore generates samples with its own code. The calculator result contains `sample_log_probs` and `sample_log_likelihoods`, so the library's estimators can still be fed with statistics collected by hand; this has not been tested on 0.7.0.

## Later stage: main run

`src/main_generate.py` runs all 1920 answerable questions: the same prompt and greedy settings as above, and 10 samples at configuration B (T = 0.7, top_k = 0, top_p = 0.95) without `min_new_tokens`, seed = 42·10⁵ + uid. It stores the token ids, the log-probabilities of every token under the model's own distribution (before temperature and top-p), the log-probability of the eos token and a flag whether generation stopped on eos. It resumes from the file on disk and takes about 1 h 55 min on a T4. `src/run_nli.py` then fills the `qa` NLI matrices for all samples (about 2 minutes on GPU). `src/main_check.py` writes `results/main_generation_check.json`.

| Check | Result |
|---|---|
| Questions, duplicates, errors | 1920, 0, 0 |
| Greedy answers equal to the first stage | 300 of 300 |
| Samples equal to the sampling-parameter stage, config B | 100 of 100 questions (1000 of 1000 samples) |
| Greedy accuracy, lemma match | 33.0% |
| Sample accuracy, lemma match | 31.1% |
| Samples without eos | 2.8% (greedy answers: 42 of 1920) |
| One-token samples | 2.2% |
| Degenerate and wrong (rule definition) | 8.05% (7.4% on the 100 questions of the parameter stage) |

The log-probability lists have the same length as the token lists in every row, and there are no NaN or positive values. The 20 questions of the calculator check were not representative: one-token answers are 2.2% over all questions against 5.0% there.

The checks before AUROC (stage A) and the H3 analysis (stage B) are described in [How to read this repository](#how-to-read-this-repository) and in `src/main_checks.md` and `src/main_results.md`.

**Not done yet.** p(True), verbalized confidence, a second model and a run on translated data, which need separate GPU runs. Manual check of about 200 answers. The corrected number and date matching is computed as a sensitivity row; the frozen `match.py` stays primary.

## Reproduce

All generation runs on Kaggle (GPU T4, Internet on) and must be started with **Save & Run All (Commit)**. An interactive session stops with the browser, and a stopped commit run loses its outputs. The analysis of the main run (steps 8 and 9) needs no GPU. Run step 8 before step 9: the plan requires the checks before any AUROC.

1. `git clone https://github.com/vladislavneon/RuBQ.git`, then `python src/step1_data.py` to get the dataset statistics (CPU only).
2. `notebooks/step2_generate.ipynb` on Kaggle (GPU T4, Internet on). Set `SMOKE=False` and use **Save & Run All (Commit)**, so that `generations.jsonl` is stored in the version output. About 17 GPU-minutes. The output is included here as `data/generations.jsonl`.
3. `notebooks/step34_metrics.ipynb` (CPU). Attach the step 2 output as input; it writes `metrics.json` and `manual_check.csv`.
4. `notebooks/step_sampling_params.ipynb`: about 14 minutes for generation and 20 for the eos check.
5. `notebooks/step_core_signals.ipynb`: self-consistency on CPU, NLI on GPU.
6. `notebooks/step_calculator_check.ipynb`: about 10 minutes.
7. `notebooks/step_main_run.ipynb`: tests, a 5-question smoke run, the full run, the check and NLI. It resumes from `data/main/generations_main.jsonl` if the file is in the repository.
8. `notebooks/step_main_checks.ipynb` (stage A, CPU): semantic entropy finiteness and agreement of the NLI and lemma groupings at configuration B. It must run before any AUROC is computed.
9. `notebooks/step_main_results.ipynb` (stage B, CPU): tests, then `src/main_results.py`. It writes the H3 comparisons, secondary metrics, cost axis and sensitivity rows to `results/main/` and the summary to `src/main_results.md`.

Tests: `python -m unittest discover -s tests`. They run on CPU with stubs for the lemmatizer and the NLI model.

## Layout

```
src/         code and notes for all stages, plus analysis_plan.md (fixed 2026-10-05, amendments in section 10)
notebooks/   one notebook per stage, run on Kaggle
tests/       unit tests (CPU, with stubs for the lemmatizer and the NLI model)
data/        RuBQ-derived inputs and generations (CC BY-SA 4.0)
results/     outputs of each stage; results/main/ holds the main run, stage A and stage B
docs/        FILE_MAP.md: every file with its stage and role
```

Code comments are partly in Russian.

## License and attribution

Code: MIT (`LICENSE`). Data in `data/` and `results/` contains questions and answers derived from **RuBQ 2.0** and is distributed under **CC BY-SA 4.0** (`data/LICENSE`).

> Rybin I., Korablinov V., Efimov P., Braslavski P. *RuBQ 2.0: An Innovated Russian Question Answering Dataset.* ESWC 2021. https://github.com/vladislavneon/RuBQ

Model outputs were generated with Qwen2.5-7B-Instruct (Apache 2.0). The NLI model is `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
