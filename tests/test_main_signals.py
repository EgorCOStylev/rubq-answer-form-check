import json, os, sys, tempfile, unittest
from unittest import mock
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import main_params as P
import main_signals as S
import main_results
import match, match_robust
import signals
from match import match_strict


def pair_matrix(texts):
    n = len(texts)
    return np.array([[texts[i] == texts[j] for j in range(n)] for i in range(n)])


class QuestionSignalsTest(unittest.TestCase):
    def test_groups_and_signals(self):
        t = ["a", "a", "b", "a", "c"]
        pm = pair_matrix(t)
        sc, se, se_w = S.question_signals(pm, pm, np.zeros(5), range(5))
        self.assertAlmostEqual(sc, 1 - 3 / 5)
        counts = np.array([3, 1, 1]) / 5
        self.assertAlmostEqual(se, float(-(np.log(counts[[0, 0, 0, 1, 2]])).sum() / 5))
        self.assertAlmostEqual(se_w, float(-(np.array([3, 1, 1]) / 5 * np.log(np.array([3, 1, 1]) / 5)).sum()))

    def test_single_group_is_zero_not_negative_zero(self):
        pm = pair_matrix(["a"] * 4)
        sc, se, se_w = S.question_signals(pm, pm, np.zeros(4), range(4))
        self.assertEqual((sc, se, se_w), (0.0, 0.0, 0.0))
        self.assertFalse(np.signbit(se))

    def test_weighted_entropy_follows_probabilities(self):
        pm = pair_matrix(["a", "b"])
        _, _, se_w = S.question_signals(pm, pm, np.array([0.0, -50.0]), range(2))
        self.assertLess(se_w, 1e-6)

    def test_no_items_is_nan(self):
        self.assertTrue(all(np.isnan(S.question_signals(np.eye(2, dtype=bool), np.eye(2, dtype=bool), np.zeros(2), []))))

    def test_extreme_log_weights_do_not_underflow(self):
        pm = pair_matrix(["a", "b"])
        _, _, se_w = S.question_signals(pm, pm, np.array([-9000.0, -9001.0]), range(2))
        self.assertTrue(np.isfinite(se_w))


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.err = rng.random(300) < 0.4
        self.u = self.err * 1.0 + rng.normal(0, 0.8, 300)
        self.u_tied = np.round(self.u)

    def test_matches_reference_auroc_with_ties(self):
        idx = np.arange(300, dtype=np.int32)[None, :]
        got = S.boot_aurocs({"a": self.u_tied}, self.err, idx)["a"][0]
        self.assertAlmostEqual(got, signals.auroc(self.u_tied, self.err))

    def test_same_seed_same_indices(self):
        self.assertTrue((S.boot_indices(50, 20, 1) == S.boot_indices(50, 20, 1)).all())

    def test_masked_path_equals_filtered_reference(self):
        valid = np.ones(300, bool)
        valid[:30] = False
        idx = S.boot_indices(300, 5, 3)
        got = S.boot_aurocs({"a": self.u}, self.err, idx, valid)["a"]
        for b in range(5):
            ii = idx[b][valid[idx[b]]]
            self.assertAlmostEqual(got[b], signals.auroc(self.u[ii], self.err[ii]))

    def test_decision_rule_boundaries(self):
        self.assertEqual(S.decide(-0.03, 0.01), "non-inferior")
        self.assertEqual(S.decide(-0.0301, 0.01), "not established within ±0.03")
        self.assertEqual(S.decide(-0.08, -0.0301), "inferior")
        self.assertEqual(S.decide(-0.08, -0.03), "not established within ±0.03")

    def test_compare_identical_signals(self):
        idx = S.boot_indices(300, 200, 2)
        c = S.compare(self.u, self.u, self.err, idx)
        self.assertEqual(c["diff"], 0.0)
        self.assertEqual(c["ci95"], [0.0, 0.0])
        self.assertEqual(c["outcome95"], "non-inferior")

    def test_compare_drops_nan_questions(self):
        a = self.u.copy()
        a[:10] = np.nan
        c = S.compare(a, self.u, self.err, S.boot_indices(300, 100, 2))
        self.assertEqual((c["n"], c["dropped"]), (290, 10))
        self.assertTrue(np.isfinite(c["ci95"]).all())

    def test_clearly_worse_signal_is_inferior(self):
        rng = np.random.default_rng(5)
        noise = rng.normal(size=300)
        c = S.compare(noise, self.u * 3, self.err, S.boot_indices(300, 300, 4))
        self.assertEqual(c["outcome95"], "inferior")


class MetricsTest(unittest.TestCase):
    def test_accuracy_at_coverage_ties(self):
        u = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], float)
        err = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 0], float)
        self.assertAlmostEqual(S.accuracy_at_coverage(u, err, 0.2), 0.5)
        self.assertAlmostEqual(S.accuracy_at_coverage(u, err, 1.0), 1 - err.mean())
        self.assertAlmostEqual(S.accuracy_at_coverage(u, err, 0.6), 1 - (2 + 2 / 3) / 6)

    def test_ece_perfect_and_worst(self):
        self.assertAlmostEqual(S.ece([0.0, 1.0], [0, 1]), 0.0)
        self.assertAlmostEqual(S.ece([1.0, 1.0], [0, 0]), 1.0)

    def test_calibration_is_out_of_fold_and_seeded(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = (x + rng.normal(size=200) > 0).astype(int)
        np.testing.assert_allclose(S.cv_calibrated(x, y), S.cv_calibrated(x, y))
        self.assertTrue(((S.cv_calibrated(x, y) > 0) & (S.cv_calibrated(x, y) < 1)).all())


def synthetic(n=80, seed=0):
    rng = np.random.default_rng(seed)
    rows, rubq, nli = [], {}, {}
    names = ["Сена", "Лаба", "Волга", "Урал", "Дон", "Нева"]
    for uid in range(n):
        gold = names[uid % len(names)]
        rubq[uid] = dict(uid=uid, answers=[dict(type="uri", label=gold, value="v", wd_names=dict(ru=[gold], en=[]), wp_names=[])])
        p_ok = 0.3 + 0.6 * rng.random()
        samples = [gold if rng.random() < p_ok else names[(uid + 1 + int(rng.integers(0, 5))) % 6] for _ in range(10)]
        greedy = gold if rng.random() < p_ok else "Иртыш"
        lps = [list(-rng.random(rng.integers(1, 5)) * (1.2 - p_ok)) for _ in range(10)]
        g = list(-rng.random(rng.integers(1, 5)) * (1.2 - p_ok))
        rows.append(dict(uid=uid, question="q", greedy=greedy, samples=samples, greedy_logprobs=g, greedy_eos_logprob=-0.1,
                         greedy_stopped_by_eos=bool(uid % 20), greedy_ntok=len(g), samples_logprobs=lps,
                         samples_ntok=[len(x) for x in lps]))
        texts = list(dict.fromkeys(samples))
        M = [[0 if a == b else 2 for b in texts] for a in texts]
        nli[uid] = dict(uid=uid, texts=texts, M=M)
    return rows, rubq, nli


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.patches = [mock.patch.object(match, "match_lemma", match_strict),
                        mock.patch.object(match_robust, "match_lemma", match_strict)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_table_and_signals(self):
        rows, rubq, nli = synthetic()
        T = S.build_table(rows, rubq, nli, match_strict)
        self.assertEqual(T["PL"].shape, (80, 10, 10))
        U = S.uncertainty_signals(T, S.sampling_signals(T))
        for k in ("comparator", "mean_lp", "sc", "se", "se_w", "length", "comparator_with_eos"):
            self.assertEqual(len(U[k]), 80)
        self.assertTrue(np.isfinite(U["se"]).all())
        keep = ~T["degen"]
        self.assertEqual(len(S.sampling_signals(T, keep=keep)["sc"]), 80)

    def test_cost_axis_endpoints(self):
        rows, rubq, nli = synthetic()
        T = S.build_table(rows, rubq, nli, match_strict)
        curve = S.cost_axis(T, T["err"], n_subsets=3)
        full = S.sampling_signals(T)
        self.assertAlmostEqual(curve["10"]["sc"]["mean"], signals.auroc(full["sc"], T["err"]))
        self.assertAlmostEqual(curve["1"]["sc"]["mean"], 0.5)
        self.assertAlmostEqual(curve["1"]["se"]["mean"], 0.5)
        self.assertEqual(curve["5"]["sc"]["n_subsets"], 3)

    def test_cost_axis_seeded(self):
        rows, rubq, nli = synthetic()
        T = S.build_table(rows, rubq, nli, match_strict)
        a = S.cost_axis(T, T["err"], n_subsets=2)
        b = S.cost_axis(T, T["err"], n_subsets=2)
        self.assertEqual(a, b)

    def test_combination_out_of_fold(self):
        rows, rubq, nli = synthetic()
        T = S.build_table(rows, rubq, nli, match_strict)
        p = S.cheap_combination_oof(T, T["err"])
        np.testing.assert_allclose(p, S.cheap_combination_oof(T, T["err"]))
        self.assertTrue(((p > 0) & (p < 1)).all())

    def test_run_writes_all_outputs(self):
        rows, rubq, nli = synthetic()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "o")
            os.makedirs(out)
            open(os.path.join(out, "deviations.md"), "w").write("# Deviations\n\n1. test\n")
            main_results.run(rows, rubq, nli, out=out, B=100, md_path=os.path.join(d, "r.md"))
            for f in ("h3.json", "secondary.json", "cost_axis.json", "sensitivity.json", "auroc_vs_K.png"):
                self.assertTrue(os.path.exists(os.path.join(out, f)), f)
            h3 = json.load(open(os.path.join(out, "h3.json")))
            self.assertIn(h3["h3_statement"], ("H3 holds", "not established within ±0.03"))
            self.assertEqual(set(h3["h3"]), {"sc_minus_comparator", "se_minus_comparator"})
            sens = json.load(open(os.path.join(out, "sensitivity.json")))
            self.assertEqual(sens["without_greedy_answers_without_eos"]["removed"], 4)
            self.assertEqual(sens["empty_sample"]["samples_with_empty_text"], 0)
            md = open(os.path.join(d, "r.md")).read()
            self.assertIn("## H3", md)
            self.assertIn("1. test", md)


if __name__ == "__main__":
    unittest.main()
