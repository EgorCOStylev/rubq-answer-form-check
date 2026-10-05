import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sp_common import load_jsonl, load_rubq, subset_ids, BASE_SEED, MAX_NEW_TOKENS, N_SAMPLES
from sp_generate import MODEL_ID, build_messages
from match import match_strict, q_kind, gold_strings


def pick_questions(rubq, rows):
    ok, bad = [], []
    for u in subset_ids():
        q = rubq[u]
        k = q_kind(q)
        hit = any(match_strict(rows[u]["greedy"], g, k) for g in gold_strings(q))
        (ok if hit else bad).append(u)
    rng = random.Random(0)
    return sorted(rng.sample(ok, 10) + rng.sample(bad, 10))


def main():
    rubq = load_rubq()
    rows = {r["uid"]: r for r in load_jsonl("data/generations.jsonl")}
    uids = pick_questions(rubq, rows)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0}).eval()
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    eos_ids = torch.tensor(eos, device=model.device)
    im_end, eot, im_start = (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"))
    report = dict(generation_config_eos=gc.eos_token_id, generation_config_pad=gc.pad_token_id,
                  tokenizer_eos=tok.eos_token_id, tokenizer_pad=tok.pad_token_id,
                  im_end=im_end, endoftext=eot, im_start=im_start, im_end_in_generation_eos=im_end in eos)
    print(json.dumps(report))

    def sample(u, min_new):
        prompt = tok.apply_chat_template(build_messages(rows[u]["question"]), tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        n_prompt = enc["input_ids"].shape[1]
        torch.manual_seed(BASE_SEED * 100000 + u)
        kw = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                  repetition_penalty=1.0, num_return_sequences=N_SAMPLES)
        if min_new:
            kw["min_new_tokens"] = min_new
        with torch.no_grad():
            out = model.generate(**enc, **kw, pad_token_id=gc.pad_token_id)
        return [x[n_prompt:] for x in out]

    out = {}
    for name, min_new in (("hf_without_min_new_tokens", 0), ("hf_min_new_tokens_2", 2)):
        tot = dict(n=0, no_eos=0, im_start_or_eot_before_im_end=0)
        examples = []
        for u in uids:
            seqs = sample(u, min_new)
            raw = [tok.decode(x, skip_special_tokens=False) for x in seqs]
            tot["n"] += len(seqs)
            tot["no_eos"] += sum(not bool(torch.isin(x, eos_ids).any()) for x in seqs)
            marked = ["<|im_start|>" in r or "<|endoftext|>" in r.split("<|im_end|>")[0] for r in raw]
            tot["im_start_or_eot_before_im_end"] += sum(marked)
            examples += [(u, r) for r, m in zip(raw, marked) if m][:2]
        out[name] = dict(totals=tot, examples=examples[:5])
        print(name, tot)

    try:
        from lm_polygraph.model_adapters import WhiteboxModel
        from lm_polygraph.utils.generation_parameters import GenerationParameters
        from lm_polygraph.stat_calculators.sample import SamplingGenerationCalculator
        wm = WhiteboxModel(model, tok, MODEL_ID, "CausalLM",
                           GenerationParameters(temperature=1.0, top_k=0, top_p=1.0, do_sample=True,
                                                repetition_penalty=1.0, allow_newlines=True), instruct=True)
        calc = SamplingGenerationCalculator(samples_n=N_SAMPLES)
        tot = dict(n=0, with_marker=0)
        for u in uids:
            torch.manual_seed(BASE_SEED * 100000 + u)
            r = calc({}, [build_messages(rows[u]["question"])], wm, max_new_tokens=MAX_NEW_TOKENS)
            for t in r["sample_texts"][0]:
                tot["n"] += 1
                tot["with_marker"] += any(m in t for m in ("<|im_start|>", "<|endoftext|>", "<|im_end|>"))
        out["lm_polygraph_calculator"] = tot
        print("lm_polygraph_calculator", tot)
    except Exception as e:
        out["lm_polygraph_calculator"] = f"failed: {type(e).__name__}: {e}"
        print(out["lm_polygraph_calculator"])
    out["report"] = report
    out["uids"] = uids
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/sampling_params_eos_check.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
