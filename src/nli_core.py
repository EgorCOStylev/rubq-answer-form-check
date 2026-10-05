import math
from collections import Counter
from match import clean
from sp_common import CYR_RX

ENT, NEU, CON = 0, 1, 2
LABELS = {"entailment": ENT, "neutral": NEU, "contradiction": CON}
VARIANTS = ("bare", "qa")


def nli_text(sample):
    return sample.split("\n")[0].strip()


def has_content(text):
    return any(ch.isalnum() for ch in text)


def unique_texts(samples):
    texts, index, idx = [], {}, []
    for s in samples:
        t = nli_text(s)
        if t not in index:
            index[t] = len(texts)
            texts.append(t)
        idx.append(index[t])
    return texts, idx


def pair_input(variant, question, a, b):
    if variant == "bare":
        return a, b
    if variant == "qa":
        return f"{question} {a}", f"{question} {b}"
    raise ValueError(variant)


def requests(variant, question, texts):
    return [(i, j) + pair_input(variant, question, a, b)
            for i, a in enumerate(texts) for j, b in enumerate(texts)
            if has_content(a) and has_content(b)]


def empty_matrix(texts):
    n = len(texts)
    return [[ENT if i == j else NEU for j in range(n)] for i in range(n)]


def fill_matrix(variant, question, texts, predict, batch=64):
    M = empty_matrix(texts)
    reqs = requests(variant, question, texts)
    for s in range(0, len(reqs), batch):
        chunk = reqs[s:s + batch]
        verdicts = predict([(p, h) for _, _, p, h in chunk])
        if len(verdicts) != len(chunk):
            raise RuntimeError("predict returned a different number of verdicts")
        for (i, j, _, _), v in zip(chunk, verdicts):
            M[i][j] = v
    return M


def same_nli(M, i, j):
    return M[i][j] == ENT and M[j][i] == ENT


def greedy_classes(n, same):
    reps, labels = [], []
    for k in range(n):
        for c, r in enumerate(reps):
            if same(k, r):
                labels.append(c)
                break
        else:
            reps.append(k)
            labels.append(len(reps) - 1)
    return labels


def se_frequency(labels):
    n = len(labels)
    cnt = Counter(labels)
    return -sum(math.log(cnt[l] / n) for l in labels) / n


def _letters(t):
    return [c for c in t if c.isalpha()]


def script_pair(a, b):
    la, lb = _letters(a), _letters(b)
    if not la or not lb:
        return False
    def kind(ls):
        if all(CYR_RX.match(c) for c in ls):
            return "cyr"
        if all(c.isascii() for c in ls):
            return "lat"
        return "mixed"
    return {kind(la), kind(lb)} == {"cyr", "lat"}


def pair_type(a, b, lemma_same, nli_same, degenerate):
    if lemma_same == nli_same:
        return None
    if lemma_same:
        if degenerate:
            return "split_degenerate"
        return "split_identical" if clean(a) == clean(b) else "split_form"
    if degenerate:
        return "merge_degenerate"
    if script_pair(a, b):
        return "merge_script_translit"
    wa, wb = set(clean(a).split()), set(clean(b).split())
    if wa and wb and (wa <= wb or wb <= wa):
        return "merge_subset"
    return "merge_other"
