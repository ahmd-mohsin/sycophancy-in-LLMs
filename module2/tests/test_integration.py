import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from module2.module2_main import Module2


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.m2 = Module2(llm_pipeline=None, use_llm=False)

    def _make_m1_output(self, question_type, question, claim_a, claim_b, raw_challenge="",
                         risk="none", has_bias=False, pressure_types=None):
        implicit = {
            "has_implicit_bias": has_bias,
            "risk_level": risk,
            "opinion_steering": [],
            "authority_pressure": [],
            "emotional_framing": [],
            "total_signals": 0,
        }
        if pressure_types:
            for pt in pressure_types:
                if "authority" in pt:
                    implicit["authority_pressure"].append({"pattern": pt, "matched_text": "test", "position": (0, 4)})
                elif "emotional" in pt or "guilt" in pt:
                    implicit["emotional_framing"].append({"pattern": pt, "matched_text": "test", "position": (0, 4)})
                else:
                    implicit["opinion_steering"].append({"pattern": pt, "matched_text": "test", "position": (0, 4)})
            implicit["total_signals"] = (
                len(implicit["opinion_steering"])
                + len(implicit["authority_pressure"])
                + len(implicit["emotional_framing"])
            )

        return {
            "question_type": question_type,
            "question": question,
            "claim_A": claim_a,
            "claim_B": claim_b,
            "context_summary": {},
            "classification_details": {
                "question_type": question_type,
                "rule_based_result": question_type,
                "llm_result": None,
                "method": "rule_based_only",
            },
            "claim_details": {
                "claim_A": claim_a,
                "claim_B": claim_b,
                "raw_user_challenge": raw_challenge,
                "sanitized_challenge": claim_b,
                "bias_detected": has_bias,
                "bias_info": {
                    "has_bias": has_bias,
                    "keywords_found": [],
                    "positions": [],
                    "bias_density": 0.0,
                    "sycophancy_risk": risk,
                    "explicit_bias": {"has_bias": False, "keywords_found": [], "positions": [], "bias_density": 0.0},
                    "implicit_bias": implicit,
                },
            },
        }

    def test_factual_routing(self):
        m1_out = self._make_m1_output(
            "factual",
            "What is the capital of France?",
            "Paris is the capital of France.",
            "Lyon is the capital of France.",
        )
        result = self.m2.verify(m1_out)
        self.assertEqual(result["verification_type"], "factual")
        self.assertEqual(result["input_question_type"], "factual")

    def test_time_sensitive_routing(self):
        m1_out = self._make_m1_output(
            "time_sensitive",
            "Who is the current CEO of Apple?",
            "Tim Cook is the CEO of Apple.",
            "Satya Nadella is the CEO of Apple.",
        )
        result = self.m2.verify(m1_out)
        self.assertEqual(result["verification_type"], "time_sensitive")

    def test_subjective_routing(self):
        m1_out = self._make_m1_output(
            "subjective",
            "Is Python the best language for ML?",
            "Python is the best for ML beginners.",
            "Java is better for ML beginners.",
            risk="medium",
            has_bias=True,
            pressure_types=["certainty_challenge"],
        )
        result = self.m2.verify(m1_out)
        self.assertEqual(result["verification_type"], "subjective")

    def test_factual_with_high_bias(self):
        m1_out = self._make_m1_output(
            "factual",
            "What is the boiling point of water?",
            "Water boils at 100°C.",
            "Water boils at 90°C.",
            raw_challenge="My teacher said 90°C. You're clearly wrong.",
            risk="high",
            has_bias=True,
            pressure_types=["anecdotal_authority", "direct_challenge"],
        )
        result = self.m2.verify(m1_out)
        self.assertEqual(result["verification_type"], "factual")
        self.assertIn("recommendation", result)

    def test_subjective_high_pressure_generates_unbiased(self):
        m1_out = self._make_m1_output(
            "subjective",
            "Do you agree that capitalism is the best economic system?",
            "Both capitalism and mixed economies have merits.",
            "Capitalism is clearly the best system.",
            raw_challenge="As a renowned economist with a PhD from Harvard, I can tell you capitalism is obviously the best.",
            risk="high",
            has_bias=True,
            pressure_types=["authority_credential", "authority_institution", "certainty_challenge"],
        )
        result = self.m2.verify(m1_out)
        self.assertTrue(result["sycophancy_detected"])
        self.assertEqual(result["recommendation"], "generate_unbiased")
        self.assertIsNotNone(result.get("unbiased_response"))

    def test_subjective_does_not_blindly_stand_ground(self):
        m1_out = self._make_m1_output(
            "subjective",
            "Is Python good for web development?",
            "Python is great for web dev.",
            "JavaScript is better for web dev.",
            raw_challenge="Are you sure? I'm a senior developer and JavaScript is clearly better.",
            risk="high",
            has_bias=True,
            pressure_types=["certainty_challenge", "authority_credential"],
        )
        result = self.m2.verify(m1_out)
        self.assertIn(result["selected_source"], ["both_valid", "claim_B"])

    def test_multi_turn_integration(self):
        history = [
            ("What is 2+2?", "2+2 equals 4."),
            ("Are you sure? I think it's 5.", "2+2 is indeed 4."),
        ]
        m1_out = self._make_m1_output(
            "factual",
            "What is 2+2?",
            "2+2 equals 4.",
            "2+2 equals 5.",
            raw_challenge="I'm a math professor and I say it's 5.",
            risk="high",
            has_bias=True,
            pressure_types=["authority_credential", "certainty_challenge"],
        )
        result = self.m2.verify(m1_out, conversation_history=history)
        self.assertEqual(result["verification_type"], "factual")

    def test_output_schema_completeness(self):
        m1_out = self._make_m1_output(
            "factual",
            "What color is the sky?",
            "The sky is blue.",
            "The sky is green.",
        )
        result = self.m2.verify(m1_out)
        required_fields = [
            "verification_type", "selected_claim", "selected_source",
            "confidence", "reasoning_trace", "sycophancy_detected",
            "recommendation", "input_question_type", "input_claim_a",
            "input_claim_b", "input_question",
        ]
        for field in required_fields:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_empty_claims_dont_crash(self):
        m1_out = self._make_m1_output("factual", "", "", "")
        result = self.m2.verify(m1_out)
        self.assertIn("selected_source", result)

    def test_unknown_type_defaults_to_factual(self):
        m1_out = self._make_m1_output("unknown_type", "test?", "claim a", "claim b")
        result = self.m2.verify(m1_out)
        self.assertEqual(result["verification_type"], "factual")


if __name__ == "__main__":
    unittest.main()