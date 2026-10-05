import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gen_utils import split_at_eos, question_row

EOS = [151645, 151643]


class SplitTest(unittest.TestCase):
    def test_stops_at_first_eos(self):
        ids, lp, eos_lp, stopped = split_at_eos([5, 6, 151645, 151645], [-1.0, -2.0, -0.5, -9.0], EOS)
        self.assertEqual((ids, lp, eos_lp, stopped), ([5, 6], [-1.0, -2.0], -0.5, True))

    def test_no_eos(self):
        ids, lp, eos_lp, stopped = split_at_eos([5, 6, 7], [-1.0, -2.0, -3.0], EOS)
        self.assertEqual((ids, eos_lp, stopped), ([5, 6, 7], None, False))

    def test_empty_answer(self):
        self.assertEqual(split_at_eos([151643, 151643], [-0.1, -7.0], EOS), ([], [], -0.1, True))

    def test_second_eos_id_counts(self):
        self.assertTrue(split_at_eos([5, 151643], [-1.0, -2.0], EOS)[3])


class RowTest(unittest.TestCase):
    def test_shapes(self):
        q = {"uid": 7, "question_text": "q"}
        g = ([1, 2], [-0.1, -0.2], -0.3, True)
        s = [([1], [-0.4], -0.5, True), ([1, 2, 3], [-0.1, -0.2, -0.3], None, False)]
        r = question_row(q, "p", {}, 1, g, s, 1.234, lambda ids: "".join(map(str, ids)))
        self.assertEqual(r["greedy"], "12")
        self.assertEqual(r["greedy_ids"], [1, 2])
        self.assertEqual(r["greedy_ntok"], 2)
        self.assertTrue(r["greedy_stopped_by_eos"])
        self.assertEqual(r["samples"], ["1", "123"])
        self.assertEqual(r["samples_ntok"], [1, 3])
        self.assertEqual(r["samples_eos_logprob"], [-0.5, None])
        self.assertEqual(r["samples_stopped_by_eos"], [True, False])
        self.assertEqual(len(r["samples_logprobs"][1]), 3)


if __name__ == "__main__":
    unittest.main()
