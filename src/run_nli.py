import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_common import load_jsonl
from nli_core import VARIANTS, LABELS, unique_texts, fill_matrix

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MAX_SEC = float(os.environ.get("NLI_MAX_SEC", 2400))
LOCK = "results/.nli_lock"


def load_predictor(device):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
    try:
        mapping = {int(i): LABELS[str(l).lower()] for i, l in model.config.id2label.items()}
    except KeyError:
        raise SystemExit(f"unexpected NLI labels: {model.config.id2label}")
    if sorted(mapping.values()) != [0, 1, 2]:
        raise SystemExit(f"unexpected NLI labels: {model.config.id2label}")

    def predict(pairs):
        enc = tok([p for p, _ in pairs], [h for _, h in pairs], return_tensors="pt",
                  padding=True, truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            idx = model(**enc).logits.argmax(-1).cpu().tolist()
        return [mapping[i] for i in idx]

    return predict


def acquire_lock():
    os.makedirs("results", exist_ok=True)
    if os.path.exists(LOCK):
        try:
            with open(LOCK) as f:
                pid = int(f.read())
            os.kill(pid, 0)
            raise SystemExit(f"another run_nli.py is active (pid {pid}); exiting")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))


def release_lock():
    if os.path.exists(LOCK):
        os.remove(LOCK)


def out_path(variant, limit):
    return f"results/nli_{'smoke_' if limit else ''}{variant}.jsonl"


def done_uids(path):
    return {r["uid"] for r in load_jsonl(path)}


def run_variant(variant, rows, predict, limit, t0):
    path = out_path(variant, limit)
    done = done_uids(path)
    todo = [r for r in rows if r["uid"] not in done]
    if not todo:
        print(f"{variant}: already complete ({len(done)} questions), skipping", flush=True)
        return True
    print(f"{variant}: {len(done)} done, {len(todo)} to do", flush=True)
    with open(path, "a") as f:
        for n, r in enumerate(todo, 1):
            if time.time() - t0 > MAX_SEC:
                print(f"{variant}: time limit reached after {n - 1} questions; rerun resumes", flush=True)
                return False
            texts, _ = unique_texts(r["samples"])
            M = fill_matrix(variant, r["question"], texts, predict)
            f.write(json.dumps(dict(uid=r["uid"], variant=variant, texts=texts, M=M), ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"{variant}: {n}/{len(todo)} questions, {time.time() - t0:.0f}s", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS) + ["both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = load_jsonl("data/generations.jsonl")
    if not rows:
        sys.exit("data/generations.jsonl is missing or empty")
    if args.limit:
        rows = rows[:args.limit]
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    if all(len(done_uids(out_path(v, args.limit))) >= len(rows) for v in variants):
        print("all requested variants are complete, nothing to do", flush=True)
        return
    acquire_lock()
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}", flush=True)
        t0 = time.time()
        predict = load_predictor(device)
        print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        ok = all([run_variant(v, rows, predict, args.limit, t0) for v in variants])
    finally:
        release_lock()
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
