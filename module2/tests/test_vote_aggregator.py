import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from module2.vote_aggregator import VoteAggregator


class TestVoteAggregator(unittest.TestCase):

    def _make_chains(self, verdicts):
        return [{"verdict": v} for v in verdicts]

    def test_unanimous_a_correct(self):
        chains_a = self._make_chains(["CORRECT", "CORRECT", "CORRECT"])
        chains_b = self._make_chains(["INCORRECT", "INCORRECT", "INCORRECT"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertEqual(result["selected_source"], "claim_A")
        self.assertEqual(result["confidence"], 1.0)

    def test_unanimous_b_correct(self):
        chains_a = self._make_chains(["INCORRECT", "INCORRECT", "INCORRECT"])
        chains_b = self._make_chains(["CORRECT", "CORRECT", "CORRECT"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertEqual(result["selected_source"], "claim_B")
        self.assertEqual(result["confidence"], 1.0)

    def test_majority_a_correct(self):
        chains_a = self._make_chains(["CORRECT", "CORRECT", "UNCERTAIN"])
        chains_b = self._make_chains(["INCORRECT", "INCORRECT", "UNCERTAIN"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertEqual(result["selected_source"], "claim_A")
        self.assertAlmostEqual(result["confidence"], 0.67, places=1)

    def test_majority_b_correct(self):
        chains_a = self._make_chains(["INCORRECT", "INCORRECT", "UNCERTAIN"])
        chains_b = self._make_chains(["CORRECT", "CORRECT", "UNCERTAIN"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertEqual(result["selected_source"], "claim_B")

    def test_conflict_both_correct(self):
        chains_a = self._make_chains(["CORRECT", "CORRECT", "UNCERTAIN"])
        chains_b = self._make_chains(["CORRECT", "CORRECT", "UNCERTAIN"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertIn(result["selected_source"], ["claim_A", "claim_B", "neither"])
        self.assertLessEqual(result["confidence"], 0.5)

    def test_all_uncertain(self):
        chains_a = self._make_chains(["UNCERTAIN", "UNCERTAIN", "UNCERTAIN"])
        chains_b = self._make_chains(["UNCERTAIN", "UNCERTAIN", "UNCERTAIN"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertEqual(result["selected_source"], "neither")
        self.assertEqual(result["confidence"], 0.0)

    def test_mixed_with_a_lead(self):
        chains_a = self._make_chains(["CORRECT", "CORRECT", "INCORRECT"])
        chains_b = self._make_chains(["CORRECT", "INCORRECT", "INCORRECT"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertEqual(result["selected_source"], "claim_A")

    def test_vote_counts(self):
        chains_a = self._make_chains(["CORRECT", "INCORRECT", "UNCERTAIN"])
        chains_b = self._make_chains(["INCORRECT", "CORRECT", "CORRECT"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertEqual(result["vote_a"]["CORRECT"], 1)
        self.assertEqual(result["vote_a"]["INCORRECT"], 1)
        self.assertEqual(result["vote_a"]["UNCERTAIN"], 1)
        self.assertEqual(result["vote_b"]["CORRECT"], 2)

    def test_vote_margin(self):
        chains_a = self._make_chains(["CORRECT", "CORRECT", "CORRECT"])
        chains_b = self._make_chains(["INCORRECT", "INCORRECT", "INCORRECT"])
        result = VoteAggregator.aggregate_votes(chains_a, chains_b)
        self.assertEqual(result["vote_margin"], 3)

    def test_count_verdicts_static(self):
        chains = [{"verdict": "CORRECT"}, {"verdict": "INCORRECT"}, {"verdict": "UNCERTAIN"}]
        counts = VoteAggregator.count_verdicts(chains)
        self.assertEqual(counts["CORRECT"], 1)
        self.assertEqual(counts["INCORRECT"], 1)
        self.assertEqual(counts["UNCERTAIN"], 1)

    def test_unknown_verdict_counted_as_uncertain(self):
        chains = [{"verdict": "MAYBE"}, {"verdict": "CORRECT"}, {"verdict": "CORRECT"}]
        counts = VoteAggregator.count_verdicts(chains)
        self.assertEqual(counts["UNCERTAIN"], 1)
        self.assertEqual(counts["CORRECT"], 2)


if __name__ == "__main__":
    unittest.main()