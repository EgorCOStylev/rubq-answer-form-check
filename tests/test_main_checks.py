import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import main_checks, analyze_nli
from match import match_strict


def row(uid, samples):
    return dict(uid=uid, samples=samples)


class SemanticEntropyCheckTest(unittest.TestCase):
    def test_counts_and_finiteness(self):
        rows = [row(1, ["a"] * 4), row(2, ["a", "b", "a", "b"])]
        nli = {1: dict(texts=["a"], M=[[0]]),
               2: dict(texts=["a", "b"], M=[[0, 2], [2, 0]])}
        res = main_checks.semantic_entropy_check(rows, nli)
        self.assertEqual(res["se_finite"], 2)
        self.assertEqual(res["se_undefined"], 0)
        self.assertEqual(res["se_zero"], 1)
        self.assertAlmostEqual(res["se_max"], np.log(2))
        self.assertEqual(res["nli_classes_hist"]["1"], 1)
        self.assertEqual(res["nli_classes_hist"]["2"], 1)

    def test_texts_mismatch_raises(self):
        with self.assertRaises(ValueError):
            main_checks.semantic_entropy_check([row(1, ["a", "b"])], {1: dict(texts=["a"], M=[[0]])})


class NoCorrectnessTest(unittest.TestCase):
    def test_question_stats_without_ok(self):
        r = dict(uid=1, greedy="a", samples=["a", "b"] * 5, samples_ntok=[1] * 10)
        q = dict(uid=1, answers=[dict(type="uri", label="a", value="v", wd_names=dict(ru=["a"], en=[]), wp_names=[])])
        texts, M = ["a", "b"], [[0, 2], [2, 0]]
        st = analyze_nli.question_stats(r, q, texts, M, match_strict, with_ok=False)
        self.assertIsNone(st["ok"])
        res = analyze_nli.summarize([st], np.random.default_rng(0))
        self.assertIsNone(res["se_mean_greedy_ok"])
        self.assertIsNone(res["se_mean_greedy_wrong"])


if __name__ == "__main__":
    unittest.main()
