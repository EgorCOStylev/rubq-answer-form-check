import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sp_common import load_jsonl, BASE_SEED, MAX_NEW_TOKENS, N_SAMPLES
from sp_generate import MODEL_ID, build_messages

UIDS_0V = [2084, 6761, 594, 620, 7045, 309, 316, 7041, 6061, 829, 6965, 194, 166, 16, 2031, 655, 2052, 8, 3016, 6330]
PROMPT_0V = ("Ответь кратко: только сущность, число или дата, без пояснений.\n"
             "Вопрос: Какая река протекает через Париж?\nОтвет: Сена\n"
             "Вопрос: В каком году основан Санкт-Петербург?\nОтвет: 1703\n"
             "Вопрос: {q}\nОтвет:")
PROMPTS = {
    "0v": lambda q: [{"role": "user", "content": PROMPT_0V.format(q=q)}],
    "0a": build_messages,
}
HF_REPEATS = 5
LMP_REPEATS = 2
MARKERS = ("<|im_start|>", "<|endoftext|>", "<|im_end|>")


def main():
    rows = {r["uid"]: r for r in load_jsonl("data/generations.jsonl")}
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0}).eval()
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    eos_ids = torch.tensor(eos, device=model.device)
    im_end, eot, im_start = (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"))
    report = dict(generation_config_eos=gc.eos_token_id, generation_config_pad=gc.pad_token_id,
                  generation_config_top_k=gc.top_k, tokenizer_eos=tok.eos_token_id, tokenizer_pad=tok.pad_token_id,
                  im_end=im_end, endoftext=eot, im_start=im_start, im_end_in_generation_eos=im_end in eos)
    print(json.dumps(report))

    def sample(messages, min_new, seed):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        n_prompt = enc["input_ids"].shape[1]
        torch.manual_seed(seed)
        kw = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                  repetition_penalty=1.0, num_return_sequences=N_SAMPLES)
        if min_new:
            kw["min_new_tokens"] = min_new
        with torch.no_grad():
            out = model.generate(**enc, **kw, pad_token_id=gc.pad_token_id)
        return [x[n_prompt:] for x in out]

    out = {}
    for pname, make in PROMPTS.items():
        for min_new in (0, 2):
            name = f"hf_prompt_{pname}_min_new_{min_new}"
            tot = dict(n=0, no_eos=0, marker_before_im_end=0, one_token_answer=0)
            examples = []
            for rep in range(HF_REPEATS):
                for u in UIDS_0V:
                    seqs = sample(make(rows[u]["question"]), min_new, BASE_SEED * 100000 + u + rep)
                    raw = [tok.decode(x, skip_special_tokens=False) for x in seqs]
                    tot["n"] += len(seqs)
                    tot["no_eos"] += sum(not bool(torch.isin(x, eos_ids).any()) for x in seqs)
                    tot["one_token_answer"] += sum(int((~torch.isin(x, eos_ids)).sum()) == 1 for x in seqs)
                    marked = ["<|im_start|>" in r or "<|endoftext|>" in r.split("<|im_end|>")[0] for r in raw]
                    tot["marker_before_im_end"] += sum(marked)
                    examples += [(u, r) for r, m in zip(raw, marked) if m][:1]
            out[name] = dict(totals=tot, examples=examples[:5])
            print(name, tot, flush=True)

    try:
        from lm_polygraph.model_adapters import WhiteboxModel
        from lm_polygraph.utils.generation_parameters import GenerationParameters
        from lm_polygraph.stat_calculators.sample import SamplingGenerationCalculator
        gp = GenerationParameters(max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, top_p=1.0, top_k=0)
        wm = WhiteboxModel(model, tok, model_path=MODEL_ID, generation_parameters=gp, instruct=True)
        calc = SamplingGenerationCalculator(samples_n=N_SAMPLES)
        for pname, make in (("0v", lambda q: PROMPT_0V.format(q=q)), ("0a", build_messages)):
            tot = dict(n=0, with_marker=0, hit_max=0)
            examples = []
            for rep in range(LMP_REPEATS):
                for u in UIDS_0V:
                    torch.manual_seed(BASE_SEED * 100000 + u + rep)
                    r = calc({}, [make(rows[u]["question"])], wm, max_new_tokens=MAX_NEW_TOKENS)
                    for t, toks in zip(r["sample_texts"][0], r["sample_tokens"][0]):
                        tot["n"] += 1
                        tot["hit_max"] += len(toks) >= MAX_NEW_TOKENS
                        if any(m in t for m in MARKERS):
                            tot["with_marker"] += 1
                            examples.append((u, t))
            out[f"lm_polygraph_calculator_prompt_{pname}"] = dict(totals=tot, examples=examples[:5])
            print("lm_polygraph_calculator", pname, tot, flush=True)
    except Exception as e:
        out["lm_polygraph_calculator"] = f"failed: {type(e).__name__}: {e}"
        print(out["lm_polygraph_calculator"])
    out["report"] = report
    out["uids"] = UIDS_0V
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/sampling_params_eos_check.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
