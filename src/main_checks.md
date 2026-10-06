# Checks before AUROC (section 9 of the analysis plan)

Data: `data/main/generations_main.jsonl`, `results/main/nli_qa.jsonl`. Script: `src/main_checks.py`, output `results/main/checks.json`. No AUROC, AURC or correctness-conditioned quantity was computed.

## Stage 0: fields

`generations_main.jsonl` (1920 rows, one per answerable question, no duplicate uid): `uid`, `question`, `prompt`, `cfg`, `gen_sample`, `seed`, `t_sec`, `greedy`, `greedy_ids`, `greedy_logprobs`, `greedy_eos_logprob`, `greedy_stopped_by_eos`, `greedy_ntok`, `samples`, `samples_ids`, `samples_logprobs`, `samples_eos_logprob`, `samples_stopped_by_eos`, `samples_ntok`.

`nli_qa.jsonl` (1920 rows, same uid order): `uid`, `variant` (all `qa`), `texts` (unique sample texts), `M` (verdict matrix, 0 entailment, 1 neutral, 2 contradiction).

Greedy log-probabilities exist for all 1920 answers. Their length equals `greedy_ntok` (1 to 16 tokens), all values are finite, the sum ranges from -14.25 to 0.0. The log-probability of eos is stored separately and is not part of `greedy_logprobs`. For all 19200 samples `ntok >= 16` coincides with `not stopped_by_eos`, so the degenerate-sample definition of the sampling-parameter stage applies as is.

## Check 1: semantic entropy

Finite on all 1920 questions, 0 undefined. Entropy is 0 on 733 questions (one NLI class), the maximum is 2.303 (ten singleton classes). Number of NLI classes per question: 1 class 733, 2 classes 290, 3 classes 200, 4 classes 134, 5 classes 125, 6 classes 115, 7 classes 106, 8 classes 80, 9 classes 78, 10 classes 59.

## Check 2: lemma grouping against NLI on sample pairs

6444 of 86400 sample pairs (1920 questions, 45 pairs each) are split differently by the lemma match and by NLI: 7.5%, 95% bootstrap interval by question 6.8% to 8.2%. At stage 0a (T = 1.0, 300 questions) the same input `qa` gave 8.5%. The value is not worse than at stage 0a, so no limitation is recorded under section 9. By answer type: entity 8.4% (1399 questions), date 8.9% (139), string 6.2% (28), number 3.5% (354). 623 questions have at least one disagreeing pair.

Types of disagreement (pairs): NLI merges what lemma keeps apart in 5567 pairs (merge_other 2510, merge_degenerate 2162, merge_subset 891, merge_script_translit 4); NLI splits what lemma joins in 877 pairs (split_form 492, split_identical 224, split_degenerate 161). The CSV `results/main/review_top50_qa.csv` lists the 50 questions with the most disagreeing pairs, with all ten samples.

Two sources of disagreement in the review file that matter for stage B:

- Number and date questions answered with a word form («Август», «IX век», «Март»). `match_lemma` applies the number or date rule, which finds no digits in a word, so even identical texts do not match. Self-consistency of these questions is maximal regardless of agreement. A count restricted to the 493 number and date questions (no lemmatizer needed): 16 questions have more lemma groups than distinct texts, in 4 of them all ten samples are one text. This is the known limitation of the frozen matching protocol; it also affects the label of the greedy answer on such questions.
- Identical texts split by NLI. The model does not return entailment on the diagonal for some texts (86 questions have at least one non-entailing diagonal), so identical samples fall into different classes: 305 identical-text pairs in 26 questions, in 2 of them (uid 175, 7049) all ten samples are one text and form ten classes, giving maximal semantic entropy.

## Check 3: questions with different class counts

The number of NLI classes and the number of lemma groups differ on 612 of 1920 questions (31.9%).
