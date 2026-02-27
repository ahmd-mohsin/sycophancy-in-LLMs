import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from module2.time_sensitive_verifier import TimeSensitiveVerifier
from module2.config import TIME_SENSITIVE_MAX_CONFIDENCE


class TestTimeSensitiveVerifier(unittest.TestCase):

    def setUp(self):
        self.verifier = TimeSensitiveVerifier(llm_pipeline=None)

    def test_basic_verification_returns_required_fields(self):
        result = self.verifier.verify(
            claim_a="Tim Cook is the CEO of Apple.",
            claim_b="Satya Nadella is the CEO of Apple.",
            question="Who is the current CEO of Apple?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertEqual(result["verification_type"], "time_sensitive")
        self.assertIn("selected_claim", result)
        self.assertIn("confidence", result)
        self.assertIn("reasoning_trace", result)
        self.assertIn("caveat", result)

    def test_confidence_capped(self):
        result = self.verifier.verify(
            claim_a="Biden is the president of the US.",
            claim_b="Trump is the president of the US.",
            question="Who is the current president of the US?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertLessEqual(result["confidence"], TIME_SENSITIVE_MAX_CONFIDENCE)

    def test_recency_warning(self):
        result = self.verifier.verify(
            claim_a="The population of Tokyo is 14 million.",
            claim_b="The population of Tokyo is 10 million.",
            question="What is the current population of Tokyo?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertTrue(result["reasoning_trace"]["recency_warning"])

    def test_source_chains_in_trace(self):
        result = self.verifier.verify(
            claim_a="Tesla stock is at $200.",
            claim_b="Tesla stock is at $400.",
            question="What is the current Tesla stock price?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        trace = result["reasoning_trace"]
        self.assertIn("source_chains_a", trace)
        self.assertIn("source_chains_b", trace)
        self.assertEqual(len(trace["source_chains_a"]), 3)

    def test_caveat_present(self):
        result = self.verifier.verify(
            claim_a="The UK prime minister is Rishi Sunak.",
            claim_b="The UK prime minister is Keir Starmer.",
            question="Who is the current UK prime minister?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertIn("caveat", result)
        self.assertTrue(len(result["caveat"]) > 0)

    def test_selected_source_values(self):
        result = self.verifier.verify(
            claim_a="The S&P 500 is at 5000.",
            claim_b="The S&P 500 is at 6000.",
            question="What is the current S&P 500 level?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertIn(result["selected_source"], ["claim_A", "claim_B", "neither"])


if __name__ == "__main__":
    unittest.main()