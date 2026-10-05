import json, os, sys, tempfile, unittest
from unittest import mock
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from match import match_strict
import signals, nli_core, run_nli, analyze_nli


def sample_aurc(u, err, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    u, err = np.asarray(u, float), np.asarray(err, float)
    vals = []
    for _ in range(n):
        order = np.lexsort((rng.random(len(u)), u))
        e = err[order]
        vals.append((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())
    return float(np.mean(vals))


class SignalsTest(unittest.TestCase):
    def setUp(self):
        self.u = [0.0, 0.0, 0.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9]
        self.err = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]

    def test_tie_aware_matches_random_tie_breaking(self):
        self.assertAlmostEqual(signals.aurc_tie_aware(self.u, self.err), sample_aurc(self.u, self.err), delta=0.003)

    def test_all_tied_equals_error_rate(self):
        self.assertAlmostEqual(signals.aurc_tie_aware([0.3] * 10, self.err), np.mean(self.err))

    def test_order_independent(self):
        perm = np.random.default_rng(0).permutation(len(self.u))
        a = signals.aurc_tie_aware(self.u, self.err)
        b = signals.aurc_tie_aware(np.array(self.u)[perm], np.array(self.err)[perm])
        self.assertAlmostEqual(a, b)

    def test_bounds_order(self):
        best, worst = signals.aurc_bounds(self.u, self.err)
        self.assertLessEqual(best, signals.aurc_tie_aware(self.u, self.err))
        self.assertLessEqual(signals.aurc_tie_aware(self.u, self.err), worst)
        self.assertLessEqual(signals.aurc_ideal(self.err), best)

    def test_auroc(self):
        self.assertEqual(signals.auroc([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(signals.auroc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5)
        self.assertTrue(np.isnan(signals.auroc([1, 2], [1, 1])))

    def test_sc_uncertainty(self):
        self.assertAlmostEqual(signals.sc_uncertainty(["a", "a", "b"], "entity", match_strict), 1 / 3, places=8)


def fake_predict(pairs):
    return [nli_core.ENT if p.split()[-1] == h.split()[-1] else nli_core.NEU for p, h in pairs]


class NliCoreTest(unittest.TestCase):
    def test_unique_texts_and_index(self):
        texts, idx = nli_core.unique_texts(["Сена\nX", "Сена", " Лаба ", ""])
        self.assertEqual(texts, ["Сена", "Лаба", ""])
        self.assertEqual(idx, [0, 0, 1, 2])

    def test_requests_skip_contentless(self):
        reqs = nli_core.requests("bare", "q", ["a", "}><", "b"])
        self.assertEqual(len(reqs), 4)
        self.assertEqual(nli_core.pair_input("qa", "q?", "a", "b"), ("q? a", "q? b"))
        with self.assertRaises(ValueError):
            nli_core.pair_input("x", "q", "a", "b")

    def test_fill_matrix_with_batches(self):
        texts = ["a", "b", "a b", "}><"]
        M = nli_core.fill_matrix("bare", "q", texts, fake_predict, batch=2)
        self.assertEqual(M[0][0], nli_core.ENT)
        self.assertEqual(M[0][1], nli_core.NEU)
        self.assertEqual(M[1][2], nli_core.ENT)
        self.assertEqual(M[3][3], nli_core.ENT)
        self.assertEqual(M[0][3], nli_core.NEU)

    def test_fill_matrix_detects_wrong_predict(self):
        with self.assertRaises(RuntimeError):
            nli_core.fill_matrix("bare", "q", ["a", "b"], lambda pairs: [0], batch=4)

    def test_classes_and_se(self):
        M = [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
        idx = [0, 1, 2, 2, 0]
        labels = nli_core.greedy_classes(5, lambda a, b: nli_core.same_nli(M, idx[a], idx[b]))
        self.assertEqual(labels, [0, 0, 1, 1, 0])
        self.assertAlmostEqual(nli_core.se_frequency([0, 1]), np.log(2))
        self.assertEqual(nli_core.se_frequency([0, 0, 0]), 0.0)

    def test_identical_text_split_when_diagonal_not_entailed(self):
        M = [[nli_core.NEU]]
        labels = nli_core.greedy_classes(2, lambda a, b: nli_core.same_nli(M, 0, 0))
        self.assertEqual(labels, [0, 1])

    def test_pair_types(self):
        t = nli_core.pair_type
        self.assertIsNone(t("a", "b", True, True, False))
        self.assertEqual(t("Афине", "Афины", True, False, False), "split_form")
        self.assertEqual(t("Сена", "сена", True, False, False), "split_identical")
        self.assertEqual(t("Сена", "сена", True, False, True), "split_degenerate")
        self.assertEqual(t("Афины", "Athens", False, True, False), "merge_script_translit")
        self.assertEqual(t("Москва", "город Москва", False, True, False), "merge_subset")
        self.assertEqual(t("Москва", "Рим", False, True, False), "merge_other")
        self.assertEqual(t("Москва", "Рим", False, True, True), "merge_degenerate")


class QuestionStatsTest(unittest.TestCase):
    def test_question_stats(self):
        q = dict(uid=1, answers=[dict(type="uri", label="Сена", wd_names=dict(ru=["Сена"], en=[]), wp_names=[], value="x")])
        r = dict(uid=1, samples=["Сена", "Сена", "Париж", "Париж", "Лондон"], samples_ntok=[1] * 5, greedy="Сена")
        texts, idx = nli_core.unique_texts(r["samples"])
        M = nli_core.empty_matrix(texts)
        for i in range(len(texts)):
            M[i][i] = nli_core.ENT
        M[1][2] = M[2][1] = nli_core.ENT
        st = analyze_nli.question_stats(r, q, texts, M, match_strict)
        self.assertEqual(st["n_pairs"], 10)
        self.assertEqual(st["lemma_classes"], 3)
        self.assertEqual(st["nli_classes"], 2)
        self.assertEqual({p["type"] for p in st["disagree"]}, {"merge_other"})
        self.assertEqual(len(st["disagree"]), 2)
        self.assertTrue(st["ok"])
        self.assertTrue(np.isfinite(st["se"]))
        with self.assertRaises(ValueError):
            analyze_nli.question_stats(r, q, ["x"], [[0]], match_strict)


class RunNliTest(unittest.TestCase):
    def setUp(self):
        self.old = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        os.makedirs("results")
        self.rows = [dict(uid=i, question="q", samples=["a", "b", "a"]) for i in range(3)]

    def tearDown(self):
        os.chdir(self.old)
        self.tmp.cleanup()

    def test_resume_and_skip(self):
        import time
        self.assertTrue(run_nli.run_variant("bare", self.rows[:2], fake_predict, 0, time.time()))
        self.assertEqual(len(run_nli.done_uids("results/nli_bare.jsonl")), 2)
        self.assertTrue(run_nli.run_variant("bare", self.rows, fake_predict, 0, time.time()))
        with open("results/nli_bare.jsonl") as f:
            lines = [json.loads(l) for l in f]
        self.assertEqual([l["uid"] for l in lines], [0, 1, 2])
        self.assertTrue(run_nli.run_variant("bare", self.rows, fake_predict, 0, time.time()))
        with open("results/nli_bare.jsonl") as f:
            self.assertEqual(len(f.readlines()), 3)

    def test_time_limit_keeps_partial_file(self):
        import time
        ok = run_nli.run_variant("qa", self.rows, fake_predict, 0, time.time() - 10 ** 6)
        self.assertFalse(ok)
        self.assertEqual(len(run_nli.done_uids("results/nli_qa.jsonl")), 0)

    def test_smoke_file_is_separate(self):
        self.assertNotEqual(run_nli.out_path("bare", 5), run_nli.out_path("bare", 0))

    def test_lock(self):
        run_nli.acquire_lock()
        with self.assertRaises(SystemExit):
            run_nli.acquire_lock()
        run_nli.release_lock()
        with open(run_nli.LOCK, "w") as f:
            f.write("999999999")
        run_nli.acquire_lock()
        run_nli.release_lock()
        self.assertFalse(os.path.exists(run_nli.LOCK))


def make_workdir():
    os.makedirs("data"), os.makedirs("results")
    rubq, rows = [], []
    names = ["Сена", "Париж", "Лондон", "Берлин", "Рим", "Мадрид"]
    for i, name in enumerate(names):
        rubq.append(dict(uid=i, answers=[dict(type="uri", label=name, value="v", wd_names=dict(ru=[name], en=[]), wp_names=[])]))
        rows.append(dict(uid=i, question=f"Вопрос {i}?", greedy=name if i % 2 == 0 else "Нью-Йорк",
                         samples=[name] * (6 + i % 2) + ["Athens", "Афины", "}><", "Рим"][: 4 - i % 2],
                         samples_ntok=[1] * 10, greedy_logprobs=[-0.1]))
    with open("data/generations.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    with open("rubq.json", "w") as f:
        json.dump(rubq, f, ensure_ascii=False)
    for v in nli_core.VARIANTS:
        with open(f"results/nli_{v}.jsonl", "w") as f:
            for r in rows:
                texts, _ = nli_core.unique_texts(r["samples"])
                M = nli_core.fill_matrix(v, r["question"], texts, fake_predict)
                f.write(json.dumps(dict(uid=r["uid"], variant=v, texts=texts, M=M), ensure_ascii=False) + "\n")


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.old = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        make_workdir()
        os.environ["RUBQ_JSON"] = "rubq.json"
        self.patch = mock.patch.object(analyze_nli, "match_lemma", match_strict)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        os.environ.pop("RUBQ_JSON", None)
        os.chdir(self.old)
        self.tmp.cleanup()

    def test_analyze_nli_main(self):
        analyze_nli.main()
        with open("results/nli_agreement.json") as f:
            res = json.load(f)
        self.assertEqual(res["bare"]["pairs"], 6 * 45)
        self.assertEqual(res["bare"]["se_finite"], 6)
        self.assertIn("ci_contains_zero", res["qa_minus_bare"])
        self.assertTrue(os.path.exists("results/review_top50_qa.csv"))

    def test_analyze_nli_incomplete_input_exits_with_message(self):
        with open("results/nli_qa.jsonl") as f:
            first = f.readline()
        with open("results/nli_qa.jsonl", "w") as f:
            f.write(first)
        with self.assertRaises(SystemExit) as cm:
            analyze_nli.main()
        self.assertIn("incomplete", str(cm.exception))

    def test_run_sc_strict(self):
        import run_sc
        sys.argv = ["run_sc.py", "--match", "strict"]
        run_sc.main()
        with open("results/sc_strict.json") as f:
            res = json.load(f)
        self.assertEqual(res["n"], 6)
        self.assertTrue(0 <= res["auroc_error"] <= 1)


if __name__ == "__main__":
    unittest.main()
