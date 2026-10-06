# Analysis plan for the main run

Status: fixed on 2026-10-06, before any AUROC or AURC was computed on the main-run data. Later changes are recorded in a dated "Amendments" section at the end of this file and are not made in place.

## 1. Scope

Data: `data/main/generations_main.jsonl` and `results/main/nli_qa.jsonl` (1920 answerable RuBQ 2.0 test questions; one greedy answer and 10 samples per question at T = 0.7, top_k = 0, top_p = 0.95; Qwen2.5-7B-Instruct, NF4).

Quantities computed on these data before this file was committed: generation checks only (counts of questions, duplicates, errors, answers without eos, one-token samples, degenerate samples; reproducibility against earlier stages). No signal was evaluated against correctness.

## 2. Information available before the plan was fixed

The following results exist from earlier stages and could have influenced the choices below.

- Stage 0a (300 of the 1920 questions, samples at T = 1.0, top_k = 0, top_p = 1.0): the share of correctness verdicts that change between strict and lemma matching (D1 = 2.2%), self-consistency AUROC against greedy error under lemma grouping (0.823), and agreement of two NLI inputs with lemma grouping (`bare` 11.3% vs `qa` 8.5% of sample pairs).
- A rough power estimate on the same 300 questions (approximate word-stem matching, entropy of answer clusters instead of NLI-based entropy, no p(True)). It included approximate AUROC values for probability-based and sampling-based signals and their point differences (sampling minus probability between -0.004 and -0.020).
- Sampling parameters were chosen on 100 of those 300 questions by the share of degenerate samples. AUROC was not used. No configuration met the pre-registered threshold; configuration B was chosen outside the rule (see `src/sampling_params.md`).

The comparator, the margin and the decision rules below were set with this information available. They were not selected by comparing AUROC values on the main-run data. The probability comparator was chosen as the more demanding of two candidates, with the stage 0a estimate in view.

## 3. Labels

The positive class is an error of the greedy answer. Correctness is determined by `match_lemma` in `src/match.py` in its frozen version. The matching protocol is not changed. Known limitations of the number and date rules (at most 8 of 300 questions at stage 0a; 26 of 1920 questions in the main run by pair counts) are recorded in `src/core_signals.md`. A corrected matching function may be added later as a robustness analysis in a separate row of results. It does not replace the main figures.

## 4. Signals

Core signals:

1. Probability comparator: sum of token log-probabilities of the greedy answer (the unnormalised sequence log-probability, equivalent to maximum sequence probability). The mean token log-probability is reported as a sensitivity analysis and is not part of the family of comparisons. Perplexity is a monotone transform of the mean and yields the same AUROC; it is not counted separately. The literature uses several single-pass probability scores (sum of token log-probabilities, length-normalised mean, mean token entropy) and names no single standard (Vashurin et al., 2025; Fadeeva et al., 2024; Alfarano et al., 2026; Kyriakou et al., 2026; Zhou et al., 2025; in Kuhn et al., 2023 the entropy baselines are computed from the same samples as semantic entropy). The sum was selected as the primary comparator because it is the more demanding of the two candidates: in the stage 0a estimate described in section 2 it scored slightly higher than the mean. The choice was made before the main-run evaluation.
2. Self-consistency: one minus the share of the largest group of samples, groups formed by lemma matching.
3. Semantic entropy: clusters of the 10 samples from NLI verdicts with the `qa` input (question plus answer); entropy of the cluster frequencies (discrete semantic entropy). The variant weighted by sample probabilities is computed as a sensitivity analysis and is not part of the family of comparisons.

Extensions run separately: p(True), verbalized confidence. Baselines: answer length, max-softmax probability.

## 5. Primary hypothesis H3

H3: at least one sampling signal (self-consistency, semantic entropy) separates correct answers from errors by AUROC not worse than the probability comparator.

Family of comparisons (m = 2): self-consistency versus comparator; semantic entropy versus comparator. For each, the difference is AUROC(sampling) minus AUROC(comparator).

Estimation: paired bootstrap over questions, the same resample for all signals, 10 000 resamples, fixed seed, percentile intervals. AUROC treats ties as 0.5.

Decision, non-inferiority margin 0.03, Bonferroni correction for m = 2: each comparison is judged at the level of a two-sided 95% interval (one-sided 2.5%), tested against the null "difference at most -0.03".

- A comparison is non-inferior if the lower limit of its 95% interval is at least -0.03.
- A comparison is inferior if the upper limit of its 95% interval is below -0.03.
- H3 holds if at least one of the two comparisons is non-inferior.
- Otherwise the outcome is reported as "not established within ±0.03", with the interval. No conclusion of equivalence is drawn from this outcome. The question of cost-effectiveness is then assessed through the cost axis (section 7).
- Intervals at the 90% level (the second step of a Holm procedure) may be reported as an additional, exploratory result. They do not enter the decision.

The margin of 0.03 is not changed after the main-run results are seen. Differences in AUROC below about 0.02-0.035 at n of about 2000 are not interpreted as a ranking of signals.

## 6. Secondary metrics

AUPRC with the base rate of errors; risk-coverage curves and AURC, tie-aware for discrete signals (the bounds for the best and worst order within ties are reported); accuracy at coverage 20%, 50% and 80%; Brier score and ECE only for quantities that are probabilities, after calibration fitted by cross-validation over questions. H1 and H2 remain auxiliary and are assessed qualitatively.

## 7. Cost axis

- AUROC of each sampling signal as a function of the number of samples K = 1…10. For K < 10, 20 random subsets of K samples per question without replacement, fixed seed; the mean over subsets is reported.
- Combination of cheap signals (sum of token log-probabilities of the greedy answer, max-softmax probability, answer length): logistic regression, 5-fold cross-validation over questions, fixed seed. The combination is compared with the best sampling signal by the same criterion (margin 0.03).

## 8. Special cases

- The 42 greedy answers without eos (truncated at 16 tokens) stay in the analysis. Sensitivity: results recomputed without them.
- Degenerate samples stay in the analysis. Sensitivity: results recomputed with degenerate samples removed (the definition of the sampling-parameter stage).
- An empty sample (question 309) forms a separate group and a separate cluster.
- If semantic entropy is undefined for a question, the question is dropped only from comparisons that involve it; the number of such questions is reported.

## 9. Checks before any AUROC is computed

- Semantic entropy is finite on all 1920 questions at configuration B.
- Agreement of NLI verdicts with lemma grouping on the samples of the main run, using the same procedure as at stage 0a. This check does not use AUROC. A value much worse than at stage 0a (T = 1.0) is reported as a limitation. The NLI input stays `qa`.

## 10. Amendments

### 2026-10-06, removal of max-softmax probability

Max-softmax probability is removed from the baselines (section 4) and from the combination of cheap signals (section 7). Under the definition of the maximum sequence probability used in LM-Polygraph (MSP) it is a monotone function of the sum of token log-probabilities of the greedy answer, so it would duplicate the primary comparator and add nothing to the combination. The baseline is answer length only. The combination in section 7 is logistic regression on the sum of token log-probabilities and answer length. No signal had been evaluated against correctness when this amendment was made. The family of comparisons, the margin and the decision rules are unchanged.
