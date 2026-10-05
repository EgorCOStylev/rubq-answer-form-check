import json, os, random, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_common import SUBSET_SEED, N_SUBSET, load_rubq
from match import q_kind

ids300 = sorted(json.load(open("data/sample_ids.json")))
uids = sorted(random.Random(SUBSET_SEED).sample(ids300, N_SUBSET))
rubq = load_rubq()
kinds = Counter(q_kind(rubq[u]) for u in uids)
out = {"seed": SUBSET_SEED, "n": N_SUBSET, "uids": uids, "kinds": dict(kinds)}
json.dump(out, open("data/sampling_params/question_ids.json", "w"), ensure_ascii=False, indent=1)
print(len(uids), dict(kinds))
