# Core signals

`signals.py` holds the self-consistency uncertainty (one minus the share of the largest group under the lemma match from `match.py`), the tie-aware AURC (exact expectation over random order inside tied scores), the naive and best/worst-order AURC, and a tie-aware AUROC.

`run_sc.py` computes the signal on `data/generations.jsonl` and writes `results/sc.json` (`results/sc_strict.json` with `--match strict`).

`nli_core.py` holds the NLI logic without torch: the two input variants (`bare`: the answers alone, `qa`: question plus answer, as in Kuhn et al., 2023), the verdict matrix over unique answer texts including the diagonal, greedy semantic classes as in LM-Polygraph 0.7.0, semantic entropy in the `frequency` form, and the types of disagreement between lemma groups and NLI classes.

`run_nli.py` fills the verdict matrices with `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` and appends one line per question to `results/nli_{variant}.jsonl`. It resumes from what is on disk, skips finished variants, refuses to start while another instance is alive, and stops after `NLI_MAX_SEC` seconds (default 2400) keeping the partial file. With `--limit N` it writes to separate `nli_smoke_*` files.

`analyze_nli.py` computes the share of pairs on which the lemma match and NLI disagree for each variant with a bootstrap interval by question, the paired difference `qa` minus `bare`, the types of disagreement, the share by answer type, the number of questions where the class counts differ, the finiteness of semantic entropy, `results/review_top50_{variant}.csv` and ten examples per variant in `results/nli_agreement.json`. A pair counts as the same under NLI when entailment holds in both directions; identical texts use the diagonal verdict.

`tests/test_core_signals.py` covers the logic with stubs for the lemmatizer and the NLI model.
