import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import match_lemma
from metrics import correct
from sp_common import load_rubq, load_jsonl, degenerate_flags, MAX_NEW_TOKENS


def main():
    rows = load_jsonl("data/main/generations_main.jsonl")
    if not rows:
        sys.exit("data/main/generations_main.jsonl is missing or empty")
    rubq = load_rubq()
    dup = [u for u, c in Counter(r["uid"] for r in rows).items() if c > 1]
    ns = sum(len(r["samples"]) for r in rows)
    one_tok = sum(n == 1 for r in rows for n in r["samples_ntok"])
    no_eos = sum(not s for r in rows for s in r["samples_stopped_by_eos"])
    degen = degen_wrong = 0
    for r in rows:
        q = rubq[r["uid"]]
        for s, stopped in zip(r["samples"], r["samples_stopped_by_eos"]):
            d = any(degenerate_flags(s, not stopped, q).values())
            degen += d
            degen_wrong += d and not correct(s, q, match_lemma)
    greedy_ok = sum(bool(correct(r["greedy"], rubq[r["uid"]], match_lemma)) for r in rows)
    out = dict(questions=len(rows), duplicate_uids=len(dup), samples=ns,
               greedy_accuracy_lemma=greedy_ok / len(rows),
               sample_accuracy_lemma=sum(bool(correct(s, rubq[r["uid"]], match_lemma)) for r in rows for s in r["samples"]) / ns,
               share_no_eos=no_eos / ns, share_one_token=one_tok / ns,
               share_degenerate_any=degen / ns, share_degenerate_and_wrong=degen_wrong / ns,
               greedy_no_eos=sum(not r["greedy_stopped_by_eos"] for r in rows),
               errors=len(load_jsonl("data/main/errors.jsonl")))

    old = {r["uid"]: r for r in load_jsonl("data/generations.jsonl")}
    same = [r["greedy"] == old[r["uid"]]["greedy"] for r in rows if r["uid"] in old]
    out["greedy_same_as_stage0a"] = dict(n=len(same), same=sum(same))

    sp = {r["uid"]: r for r in load_jsonl("data/sampling_params/generations_sp.jsonl") if r["cfg"] == "B"}
    pairs = [(r["samples"], sp[r["uid"]]["samples"]) for r in rows if r["uid"] in sp]
    out["samples_same_as_sampling_params_B"] = dict(
        n=len(pairs), identical_lists=sum(a == b for a, b in pairs),
        identical_samples=sum(x == y for a, b in pairs for x, y in zip(a, b)), total_samples=10 * len(pairs))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/main_generation_check.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
