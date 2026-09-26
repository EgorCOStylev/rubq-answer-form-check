# Does answer-form variation matter in Russian short-answer QA? A pre-registered check on RuBQ 2.0

**TL;DR — negative result.** With a short-answer prompt, Qwen2.5-7B-Instruct (4-bit) almost never answers in an inflected or prepositional form. Lemmatized matching changes the correctness verdict for **2.2%** of answers and the self-consistency score for **1.7%** of questions. Both are below the 3% threshold fixed before the experiment, so by the pre-registered rule the premise is rejected.

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

## Reproduce

1. `git clone https://github.com/vladislavneon/RuBQ.git`, then `python src/step1_data.py` to get the dataset statistics (CPU only).
2. `notebooks/step2_generate.ipynb` on Kaggle (GPU T4, Internet on). Set `SMOKE=False` and use **Save & Run All (Commit)**, so that `generations.jsonl` is stored in the version output. About 17 GPU-minutes. The output is included here as `data/generations.jsonl`.
3. `notebooks/step34_metrics.ipynb` (CPU). Attach the step 2 output as input; it writes `metrics.json` and `manual_check.csv`.

## Layout

```
src/        step1_data.py, match.py, metrics.py
notebooks/  step2_generate.ipynb, step34_metrics.ipynb
data/       sample_ids.json, generations.jsonl   (CC BY-SA 4.0, derived from RuBQ 2.0)
results/    metrics.json, manual_check.csv
```

Code comments are partly in Russian.

## License and attribution

Code: MIT (`LICENSE`). Data in `data/` and `results/` contains questions and answers derived from **RuBQ 2.0** and is distributed under **CC BY-SA 4.0** (`data/LICENSE`).

> Rybin I., Korablinov V., Efimov P., Braslavski P. *RuBQ 2.0: An Innovated Russian Question Answering Dataset.* ESWC 2021. https://github.com/vladislavneon/RuBQ

Model outputs were generated with Qwen2.5-7B-Instruct (Apache 2.0).
