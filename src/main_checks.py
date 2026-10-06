"""Checks of section 9 of src/analysis_plan.md on the main-run data. No AUROC, no AURC, no correctness."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_common import load_jsonl
from nli_core import same_nli, greedy_classes, se_frequency, unique_texts
import analyze_nli

DATA = "data/main/generations_main.jsonl"
NLI_DIR = "results/main"
OUT = "results/main"
STAGE_0A_QA_SHARE, STAGE_0A_BARE_SHARE = 0.085, 0.113


def semantic_entropy_check(rows, nli):
    se, n_classes = [], []
    for r in rows:
        d = nli[r["uid"]]
        texts, idx = unique_texts(r["samples"])
        if texts != d["texts"]:
            raise ValueError(f"uid {r['uid']}: stored NLI texts do not match the samples")
        labels = greedy_classes(len(r["samples"]), lambda a, b: same_nli(d["M"], idx[a], idx[b]))
        se.append(se_frequency(labels))
        n_classes.append(len(set(labels)))
    se = np.array(se)
    return dict(questions=len(rows), se_finite=int(np.isfinite(se).sum()), se_undefined=int((~np.isfinite(se)).sum()),
                se_zero=int((se == 0).sum()), se_min=float(np.nanmin(se)), se_max=float(np.nanmax(se)),
                nli_classes_hist={str(k): int((np.array(n_classes) == k).sum()) for k in range(1, 11)})


def main(lemma_free=False):
    rows = load_jsonl(DATA)
    nli = {r["uid"]: r for r in load_jsonl(f"{NLI_DIR}/nli_qa.jsonl")}
    out = dict(check_1_semantic_entropy=semantic_entropy_check(rows, nli))
    if lemma_free:
        json.dump(out, open(f"{OUT}/checks_lemma_free.json", "w"), indent=1)
        print(json.dumps(out, indent=1))
        return out
    res, _ = analyze_nli.main(["--data", DATA, "--nli-dir", NLI_DIR, "--out-dir", OUT, "--variants", "qa",
                               "--no-correctness"])
    q = res["qa"]
    out["check_2_lemma_vs_nli_pairs"] = dict(
        pairs=q["pairs"], disagree=q["disagree"], share=q["share"], share_ci95_bootstrap_by_question=q["share_ci"],
        types=q["types"], by_answer_type=q["by_kind"], questions_with_disagreement=q["questions_with_disagreement"],
        lemma_same_nli_split=q["lemma_same_nli_split"], nli_same_lemma_split=q["nli_same_lemma_split"],
        review_csv=f"{OUT}/review_top50_qa.csv", examples=q["examples"],
        stage_0a_reference=dict(qa=STAGE_0A_QA_SHARE, bare=STAGE_0A_BARE_SHARE))
    out["check_3_class_counts"] = dict(questions_class_count_differs=q["questions_class_count_differs"],
                                       questions=len(rows))
    json.dump(out, open(f"{OUT}/checks.json", "w"), ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemma-free", action="store_true")
    main(ap.parse_args().lemma_free)
