def split_at_eos(ids, logprobs, eos_ids):
    """Cut a generated row at its first eos token. Returns (ids, logprobs, eos_logprob, stopped)."""
    eos = set(eos_ids)
    for i, t in enumerate(ids):
        if t in eos:
            return ids[:i], logprobs[:i], logprobs[i], True
    return list(ids), list(logprobs), None, False


def question_row(q, prompt, gen_sample, seed, greedy, samples, dt, tok_decode):
    """greedy and samples are lists of (ids, logprobs, eos_logprob, stopped) tuples."""
    def block(prefix, items):
        return {
            f"{prefix}_ids": [x[0] for x in items],
            f"{prefix}_logprobs": [[round(v, 5) for v in x[1]] for x in items],
            f"{prefix}_eos_logprob": [None if x[2] is None else round(x[2], 5) for x in items],
            f"{prefix}_stopped_by_eos": [x[3] for x in items],
            f"{prefix}_ntok": [len(x[0]) for x in items],
        }
    g = block("greedy", [greedy])
    row = {"uid": q["uid"], "question": q["question_text"], "prompt": prompt, "cfg": "B",
           "gen_sample": gen_sample, "seed": seed,
           "greedy": tok_decode(greedy[0]), "samples": [tok_decode(x[0]) for x in samples],
           "t_sec": round(dt, 3)}
    row.update({k: v[0] for k, v in g.items()})
    row.update(block("samples", samples))
    return row
