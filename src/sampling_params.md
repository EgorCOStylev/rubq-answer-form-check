# Sampling parameters

Choice of temperature, top_k and top_p for the main run. Four configurations are compared on 100 questions drawn with seed 20261004 from the 300 of the answer-form check (`data/sampling_params/question_ids.json`). Qwen2.5-7B-Instruct, 4-bit, 10 samples per question, same prompt and `max_new_tokens = 16` as in `step2_generate.ipynb`.

| | T | top_k | top_p |
|---|---|---|---|
| A | 1.0 | 0 | 1.0 |
| B | 0.7 | 0 | 0.95 |
| C | 1.0 | 50 | 0.95 |
| D | 0.8 | 50 | 0.95 |

A is taken from `data/generations.jsonl` and is not regenerated. The greedy answer does not depend on sampling parameters and is taken from the same file.

## Selection rule

Fixed before generation. A sample is degenerate if it is wrong under the lemma matching of `match.py` and at least one of the following holds: it reached `max_new_tokens` without `<|im_end|>`; it contains CJK characters; the reference and its aliases contain Cyrillic while the share of Cyrillic among the letters of the answer is below 50%. The same definition applied to greedy answers on the same 100 questions gives the floor. A configuration is dropped if its degenerate share exceeds the floor by more than 2 pp. Among the rest the one with the highest sample accuracy wins. If several are within 1 pp of the best, the one with the smaller |T - 1| wins. AUROC is not used. If nothing passes, the run stops and the grid is not extended.

## Files

`sp_select.py` draws the questions. `sp_generate.py` generates B, C and D into `data/sampling_params/generations_sp.jsonl`, one block per question, and resumes from what is already on disk. `sp_check_eos.py` checks the stop condition on the 20 questions of the LM-Polygraph check (stage 0v): eos ids in the model config, and the effect of `min_new_tokens=2` that LM-Polygraph 0.7.0 sets in its sampling calculator, with the stage 0v prompt and with the stage 0a prompt. `sp_metrics.py` computes the metrics and applies the rule. `sp_common.py` holds the constants and the degeneracy definition.

## Running

On Kaggle with a T4, run `notebooks/step_sampling_params.ipynb` with Save & Run All. Generation takes about 14 minutes and the eos check about 20, each preceded by a model load of a few minutes. Outputs are `data/sampling_params/generations_sp.jsonl`, `results/sampling_params_metrics.json` and `results/sampling_params_eos_check.json`.
