import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sp_common import CONFIGS, GENERATED, BASE_SEED, MAX_NEW_TOKENS, N_SAMPLES, subset_ids, load_jsonl

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUT = os.environ.get("SP_OUT", "data/sampling_params/generations_sp.jsonl")
MAX_GEN_SEC = float(os.environ.get("SP_MAX_GEN_SEC", 2700))

SYSTEM = ("Ты отвечаешь на вопросы викторины. Ответь кратко: только сущность, число или дата. "
          "Без пояснений, без полного предложения.")
FEWSHOT = [
    ("Какой город является столицей Японии?", "Токио"),
    ("Сколько планет в Солнечной системе?", "8"),
    ("Кто написал роман «Преступление и наказание»?", "Фёдор Достоевский"),
    ("Когда был основан Санкт-Петербург?", "1703"),
]


def build_messages(question):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in FEWSHOT:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": question})
    return msgs


def main():
    a_rows = {r["uid"]: r for r in load_jsonl("data/generations.jsonl")}
    uids = subset_ids()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0}).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    eos_ids = torch.tensor(eos, device=model.device)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    print("eos:", eos, "tokenizer eos:", tok.eos_token_id, "im_end:", im_end, "pad:", tok.pad_token_id)
    assert im_end in eos and tok.eos_token_id == im_end

    def prompt_of(row):
        return tok.apply_chat_template(build_messages(row["question"]), tokenize=False, add_generation_prompt=True)

    for u in uids:
        assert prompt_of(a_rows[u]) == a_rows[u]["prompt"], f"prompt mismatch uid={u}"

    def run(row, cfg):
        enc = tok(prompt_of(row), return_tensors="pt").to(model.device)
        n_prompt = enc["input_ids"].shape[1]
        p = CONFIGS[cfg]
        gen = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=p["temperature"], top_k=p["top_k"],
                   top_p=p["top_p"], repetition_penalty=1.0, num_return_sequences=N_SAMPLES)
        seed = BASE_SEED * 100000 + row["uid"]
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, **gen, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
        ids = [x[n_prompt:] for x in out]
        return {
            "cfg": cfg, "uid": row["uid"], "question": row["question"], "gen_sample": gen, "seed": seed,
            "samples": [tok.decode(x, skip_special_tokens=True).strip() for x in ids],
            "samples_raw": [tok.decode(x, skip_special_tokens=False) for x in ids],
            "samples_ntok": [int((~torch.isin(x, eos_ids)).sum()) for x in ids],
            "stopped_by_eos": [bool(torch.isin(x, eos_ids).any()) for x in ids],
            "t_sec": round(dt, 3),
        }

    done = {(r["cfg"], r["uid"]) for r in load_jsonl(OUT)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    run(a_rows[uids[0]], "B")  # CUDA warm-up, not saved
    t_start = time.time()
    n = 0
    with open(OUT, "a") as f:
        for i, u in enumerate(uids):
            todo = [c for c in GENERATED if (c, u) not in done]
            if not todo:
                continue
            if time.time() - t_start > MAX_GEN_SEC:
                print(f"time limit reached at question {i}/{len(uids)}")
                break
            for c in todo:
                f.write(json.dumps(run(a_rows[u], c), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if i % 10 == 0:
                print(i, f"{time.time() - t_start:.0f}s", flush=True)
    print("questions:", n, "sec:", round(time.time() - t_start))


if __name__ == "__main__":
    main()
