# Checks before AUROC (section 9 of the analysis plan)

Data: `data/main/generations_main.jsonl`, `results/main/nli_qa.jsonl`. Script: `src/main_checks.py`, output `results/main/checks.json`. No AUROC, AURC or correctness-conditioned quantity was computed.

## Stage 0: fields

`generations_main.jsonl` (1920 rows, one per answerable question, no duplicate uid): `uid`, `question`, `prompt`, `cfg`, `gen_sample`, `seed`, `t_sec`, `greedy`, `greedy_ids`, `greedy_logprobs`, `greedy_eos_logprob`, `greedy_stopped_by_eos`, `greedy_ntok`, `samples`, `samples_ids`, `samples_logprobs`, `samples_eos_logprob`, `samples_stopped_by_eos`, `samples_ntok`.

`nli_qa.jsonl` (1920 rows, same uid order): `uid`, `variant` (all `qa`), `texts` (unique sample texts), `M` (verdict matrix, 0 entailment, 1 neutral, 2 contradiction).

Greedy log-probabilities exist for all 1920 answers. Their length equals `greedy_ntok` (1 to 16 tokens), all values are finite, the sum ranges from -14.25 to 0.0. The log-probability of eos is stored separately and is not part of `greedy_logprobs`. For all 19200 samples `ntok >= 16` coincides with `not stopped_by_eos`, so the degenerate-sample definition of the sampling-parameter stage applies as is.

## Check 1: semantic entropy

Finite on all 1920 questions, 0 undefined. Entropy is 0 on 733 questions (one NLI class), the maximum is 2.303 (ten singleton classes). Number of NLI classes per question: 1 class 733, 2 classes 290, 3 classes 200, 4 classes 134, 5 classes 125, 6 classes 115, 7 classes 106, 8 classes 80, 9 classes 78, 10 classes 59.

## Check 2: lemma grouping against NLI on sample pairs

Not computed yet: `match_lemma` needs `pymorphy3`, which could not be installed in the session that prepared the code (see `results/main/deviations.md`). Reference at stage 0a: 8.5% (`qa`), 11.3% (`bare`).

## Check 3: questions with different class counts

Not computed yet, same reason.
