import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from module2.factual_verifier import FactualVerifier


class TestFactualVerifier(unittest.TestCase):

    def setUp(self):
        self.verifier = FactualVerifier(llm_pipeline=None)

    def test_basic_verification_returns_required_fields(self):
        result = self.verifier.verify(
            claim_a="Paris is the capital of France.",
            claim_b="Lyon is the capital of France.",
            question="What is the capital of France?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertIn("verification_type", result)
        self.assertIn("selected_claim", result)
        self.assertIn("selected_source", result)
        self.assertIn("confidence", result)
        self.assertIn("reasoning_trace", result)
        self.assertIn("sycophancy_detected", result)
        self.assertIn("recommendation", result)
        self.assertEqual(result["verification_type"], "factual")

    def test_reasoning_trace_has_chains(self):
        result = self.verifier.verify(
            claim_a="Water boils at 100°C.",
            claim_b="Water boils at 90°C.",
            question="What is the boiling point of water?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        trace = result["reasoning_trace"]
        self.assertIn("claim_a_chains", trace)
        self.assertIn("claim_b_chains", trace)
        self.assertEqual(len(trace["claim_a_chains"]), 3)
        self.assertEqual(len(trace["claim_b_chains"]), 3)

    def test_chain_perspectives(self):
        result = self.verifier.verify(
            claim_a="2 + 2 = 4",
            claim_b="2 + 2 = 5",
            question="What is 2 + 2?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        perspectives = [c["perspective"] for c in result["reasoning_trace"]["claim_a_chains"]]
        self.assertIn("analytical", perspectives)
        self.assertIn("adversarial", perspectives)
        self.assertIn("knowledge_based", perspectives)

    def test_vote_counts_in_trace(self):
        result = self.verifier.verify(
            claim_a="The Earth orbits the Sun.",
            claim_b="The Sun orbits the Earth.",
            question="Does the Earth orbit the Sun?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        trace = result["reasoning_trace"]
        self.assertIn("vote_a", trace)
        self.assertIn("vote_b", trace)
        self.assertIn("vote_margin", trace)
        for key in ["CORRECT", "INCORRECT", "UNCERTAIN"]:
            self.assertIn(key, trace["vote_a"])
            self.assertIn(key, trace["vote_b"])

    def test_confidence_range(self):
        result = self.verifier.verify(
            claim_a="Tokyo is in Japan.",
            claim_b="Tokyo is in China.",
            question="What country is Tokyo in?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_selected_source_values(self):
        result = self.verifier.verify(
            claim_a="Pi is approximately 3.14159.",
            claim_b="Pi is approximately 3.0.",
            question="What is the value of pi?",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertIn(result["selected_source"], ["claim_A", "claim_B", "neither"])

    def test_recommendation_values(self):
        result = self.verifier.verify(
            claim_a="The speed of light is 299,792,458 m/s.",
            claim_b="The speed of light is 300,000 m/s.",
            question="What is the speed of light?",
            bias_info={"has_bias": True, "sycophancy_risk": "high"},
        )
        self.assertIn(result["recommendation"], ["maintain_original", "accept_correction", "flag_uncertain"])

    def test_sycophancy_flag_with_bias(self):
        result = self.verifier.verify(
            claim_a="Mount Everest is the tallest mountain.",
            claim_b="K2 is the tallest mountain.",
            question="What is the tallest mountain?",
            bias_info={"has_bias": True, "sycophancy_risk": "high"},
        )
        if result["selected_source"] == "claim_A":
            self.assertTrue(result["sycophancy_detected"])

    def test_empty_claims(self):
        result = self.verifier.verify(
            claim_a="",
            claim_b="",
            question="",
            bias_info={"has_bias": False, "sycophancy_risk": "none"},
        )
        self.assertIn("selected_source", result)


if __name__ == "__main__":
    unittest.main()