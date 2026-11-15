"""Fix the question subsets used after main-results-v1, before any of them is used.

Writes data/subsets/:
  manual_check_uids.json  200 uids for the manual check of the automatic labels
  threshold_split.json    960 calibration and 960 holdout uids for the abstention threshold
  config_a_check_uids.json  500 uids for the check that the H3 conclusion does not depend on configuration B

Each subset is drawn independently from the 1920 uids of the main run with its own seed.
No labels or signals are read, so the draw cannot depend on results.
"""
import json, os, random

DATA = "data/main/generations_main.jsonl"
OUT = "data/subsets"
SEEDS = {"manual_check": 20261115, "threshold_split": 20261116, "config_a_check": 20261117}


def main():
    uids = sorted(json.loads(l)["uid"] for l in open(DATA))
    assert len(uids) == 1920 and len(set(uids)) == 1920
    os.makedirs(OUT, exist_ok=True)

    def dump(name, obj):
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)

    s = SEEDS["manual_check"]
    dump("manual_check_uids.json", {"seed": s, "n": 200, "source": DATA,
                                    "uids": sorted(random.Random(s).sample(uids, 200))})

    s = SEEDS["threshold_split"]
    sh = uids[:]
    random.Random(s).shuffle(sh)
    dump("threshold_split.json", {"seed": s, "source": DATA,
                                  "calibration": sorted(sh[:960]), "holdout": sorted(sh[960:])})

    s = SEEDS["config_a_check"]
    dump("config_a_check_uids.json", {"seed": s, "n": 500, "source": DATA,
                                      "config": {"name": "A", "temperature": 1.0, "top_k": 0, "top_p": 1.0},
                                      "uids": sorted(random.Random(s).sample(uids, 500))})


if __name__ == "__main__":
    main()
