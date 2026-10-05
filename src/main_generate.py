import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sp_common import CONFIGS, BASE_SEED, MAX_NEW_TOKENS, N_SAMPLES, load_rubq, load_jsonl
from sp_generate import MODEL_ID, build_messages
from gen_utils import split_at_eos, question_row

CFG = CONFIGS["B"]
MAX_SEC = float(os.environ.get("MAIN_MAX_SEC", 3 * 3600))
MAX_ERRORS = 20
GREEDY = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1, repetition_penalty=1.0,
              temperature=None, top_p=None, top_k=None)
SAMPLE = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, repetition_penalty=1.0,
              num_return_sequences=N_SAMPLES, **CFG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = "data/main/generations_main_smoke.jsonl" if args.limit else "data/main/generations_main.jsonl"
    err_path = "data/main/errors.jsonl"
    os.makedirs("data/main", exist_ok=True)

    questions = sorted((q for q in load_rubq().values() if q["answers"]), key=lambda q: q["uid"])
    assert len(questions) == 1920, len(questions)
    if args.limit:
        questions = questions[:args.limit]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0}).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    assert tok.convert_tokens_to_ids("<|im_end|>") in eos and tok.eos_token_id in eos
    print("eos:", eos, "device:", model.device, flush=True)

    def prompt_of(q):
        return tok.apply_chat_template(build_messages(q["question_text"]), tokenize=False, add_generation_prompt=True)

    ref = {r["uid"]: r for r in load_jsonl("data/generations.jsonl")}
    for q in questions:
        if q["uid"] in ref:
            assert prompt_of(q) == ref[q["uid"]]["prompt"], f"prompt differs from stage 0a, uid={q['uid']}"

    def generate(enc, n_prompt, kw):
        with torch.no_grad():
            g = model.generate(**enc, **kw, output_logits=True, return_dict_in_generate=True,
                               pad_token_id=tok.eos_token_id)
        lp = model.compute_transition_scores(g.sequences, g.logits, normalize_logits=True).float().cpu().tolist()
        ids = g.sequences[:, n_prompt:].cpu().tolist()
        return [split_at_eos(i, l, eos) for i, l in zip(ids, lp)]

    def run(q):
        prompt = prompt_of(q)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        n_prompt = enc["input_ids"].shape[1]
        seed = BASE_SEED * 100000 + q["uid"]
        torch.cuda.synchronize()
        t0 = time.time()
        greedy = generate(enc, n_prompt, GREEDY)[0]
        torch.manual_seed(seed)
        samples = generate(enc, n_prompt, SAMPLE)
        torch.cuda.synchronize()
        dec = lambda ids: tok.decode(ids, skip_special_tokens=True).strip()
        return question_row(q, prompt, SAMPLE, seed, greedy, samples, time.time() - t0, dec)

    done = {r["uid"] for r in load_jsonl(out)}
    todo = [q for q in questions if q["uid"] not in done]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    if todo:
        run(todo[0])  # CUDA warm-up, not saved
    n_err = 0
    t_start = time.time()
    finished = True
    with open(out, "a") as f, open(err_path, "a") as ferr:
        for n, q in enumerate(todo, 1):
            if time.time() - t_start > MAX_SEC:
                print(f"time limit reached after {n - 1} questions; rerun resumes", flush=True)
                finished = False
                break
            try:
                f.write(json.dumps(run(q), ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                n_err += 1
                ferr.write(json.dumps(dict(uid=q["uid"], error=f"{type(e).__name__}: {e}")) + "\n")
                ferr.flush()
                torch.cuda.empty_cache()
                print(f"uid {q['uid']} failed: {type(e).__name__}: {e}", flush=True)
                if n_err > MAX_ERRORS:
                    sys.exit("too many errors")
            if n % 25 == 0:
                el = time.time() - t_start
                print(f"{n}/{len(todo)} questions, {el:.0f}s, eta {el / n * (len(todo) - n):.0f}s", flush=True)
    if not finished or n_err:
        sys.exit(2)


if __name__ == "__main__":
    main()
