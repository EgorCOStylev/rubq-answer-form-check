import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sp_common import load_jsonl, BASE_SEED, MAX_NEW_TOKENS, N_SAMPLES, CONFIGS
from sp_generate import MODEL_ID, build_messages
from sp_check_eos import UIDS_0V, MARKERS

REPEATS = 3
CFG = CONFIGS["B"]


def main():
    rows = {r["uid"]: r for r in load_jsonl("data/generations.jsonl")}
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0}).eval()
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    eos_ids = torch.tensor(eos, device=model.device)
    out = dict(config=CFG, uids=UIDS_0V, repeats=REPEATS)

    def hf_sample(question, seed):
        prompt = tok.apply_chat_template(build_messages(question), tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        n_prompt = enc["input_ids"].shape[1]
        torch.manual_seed(seed)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, repetition_penalty=1.0,
                               num_return_sequences=N_SAMPLES, pad_token_id=gc.pad_token_id, **CFG)
        return [x[n_prompt:] for x in o]

    tot = dict(n=0, no_eos=0, one_token_answer=0)
    for rep in range(REPEATS):
        for u in UIDS_0V:
            seqs = hf_sample(rows[u]["question"], BASE_SEED * 100000 + u + rep)
            tot["n"] += len(seqs)
            tot["no_eos"] += sum(not bool(torch.isin(x, eos_ids).any()) for x in seqs)
            tot["one_token_answer"] += sum(int((~torch.isin(x, eos_ids)).sum()) == 1 for x in seqs)
    out["hf_generate_B"] = tot
    print("hf_generate_B", tot, flush=True)

    try:
        from lm_polygraph.model_adapters import WhiteboxModel
        from lm_polygraph.utils.generation_parameters import GenerationParameters
        from lm_polygraph.stat_calculators.sample import SamplingGenerationCalculator
        gp = GenerationParameters(max_new_tokens=MAX_NEW_TOKENS, **CFG)
        wm = WhiteboxModel(model, tok, model_path=MODEL_ID, generation_parameters=gp, instruct=True)
        calc = SamplingGenerationCalculator(samples_n=N_SAMPLES)
    except Exception as e:
        out["calculator"] = f"setup failed: {type(e).__name__}: {e}"
        print(out["calculator"], flush=True)
        calc = None

    if calc is not None:
        tot = dict(n=0, with_marker=0, hit_max=0, one_token_answer=0, errors=0)
        examples, keys, lens = [], None, []
        for rep in range(REPEATS):
            for u in UIDS_0V:
                torch.manual_seed(BASE_SEED * 100000 + u + rep)
                try:
                    r = calc({}, [build_messages(rows[u]["question"])], wm, max_new_tokens=MAX_NEW_TOKENS)
                except Exception as e:
                    tot["errors"] += 1
                    out.setdefault("first_error", f"{type(e).__name__}: {e}")
                    continue
                keys = keys or sorted(r.keys())
                for t, toks in zip(r["sample_texts"][0], r["sample_tokens"][0]):
                    toks = [x for x in toks if x not in eos]
                    tot["n"] += 1
                    lens.append(len(toks))
                    tot["hit_max"] += len(toks) >= MAX_NEW_TOKENS
                    tot["one_token_answer"] += len(toks) == 1
                    if any(m in t for m in MARKERS):
                        tot["with_marker"] += 1
                        examples.append((u, t))
        out["calculator"] = dict(totals=tot, result_keys=keys, examples=examples[:5],
                                 min_len=min(lens) if lens else None)
        print("calculator", out["calculator"], flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/calculator_check.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
