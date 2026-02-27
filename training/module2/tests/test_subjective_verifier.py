import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from module2.subjective_verifier import SubjectiveVerifier


class TestSubjectiveVerifier(unittest.TestCase):

    def setUp(self):
        self.verifier = SubjectiveVerifier(llm_pipeline=None)

    def _make_bias_info(self, risk="none", has_bias=False, pressure_types=None):
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
                if pt in ("opinion_steering", "certainty_challenge", "appeal_to_prior_belief"):
                    implicit["opinion_steering"].append({"pattern": pt, "matched_text": "test", "position": (0, 4)})
                elif pt in ("authority_credential", "authority_institution", "authority_experience"):
                    implicit["authority_pressure"].append({"pattern": pt, "matched_text": "test", "position": (0, 4)})
                elif pt in ("emotional_intensifier", "guilt_tripping"):
                    implicit["emotional_framing"].append({"pattern": pt, "matched_text": "test", "position": (0, 4)})
            implicit["total_signals"] = (
                len(implicit["opinion_steering"])
                + len(implicit["authority_pressure"])
                + len(implicit["emotional_framing"])
            )

        return {
            "has_bias": has_bias,
            "keywords_found": [],
            "positions": [],
            "bias_density": 0.0,
            "sycophancy_risk": risk,
            "explicit_bias": {"has_bias": False, "keywords_found": [], "positions": [], "bias_density": 0.0},
            "implicit_bias": implicit,
        }

    def test_basic_fields(self):
        result = self.verifier.verify(
            claim_a="Python is best for ML beginners.",
            claim_b="Java is best for ML beginners.",
            bias_info=self._make_bias_info(),
        )
        self.assertEqual(result["verification_type"], "subjective")
        self.assertIn("selected_claim", result)
        self.assertIn("confidence", result)
        self.assertIn("sycophancy_detected", result)
        self.assertIn("unbiased_response", result)
        self.assertIn("suggested_response_strategy", result)

    def test_high_pressure_detects_sycophancy(self):
        result = self.verifier.verify(
            claim_a="Python is best for ML.",
            claim_b="Java is best for ML.",
            bias_info=self._make_bias_info(
                risk="high",
                has_bias=True,
                pressure_types=["certainty_challenge", "authority_credential"],
            ),
            raw_challenge="Are you sure? I'm a professor at Stanford with 20 years experience.",
        )
        self.assertTrue(result["sycophancy_detected"])
        self.assertEqual(result["recommendation"], "generate_unbiased")

    def test_high_pressure_generates_unbiased_output(self):
        result = self.verifier.verify(
            claim_a="Python is best for ML.",
            claim_b="Java is best for ML.",
            bias_info=self._make_bias_info(
                risk="high",
                has_bias=True,
                pressure_types=["certainty_challenge", "authority_credential"],
            ),
            raw_challenge="Are you sure? I'm a professor at Stanford with 20 years experience.",
        )
        self.assertIsNotNone(result["unbiased_response"])
        self.assertIn("claim_a_merit", result["unbiased_response"])
        self.assertIn("claim_b_merit", result["unbiased_response"])

    def test_sycophancy_does_not_just_stand_ground(self):
        result = self.verifier.verify(
            claim_a="React is the best framework.",
            claim_b="Vue is the best framework.",
            bias_info=self._make_bias_info(
                risk="high",
                has_bias=True,
                pressure_types=["certainty_challenge", "authority_credential"],
            ),
            raw_challenge="Are you sure? I'm a senior engineer and Vue is clearly better.",
        )
        self.assertNotEqual(result["selected_source"], "claim_A")
        self.assertIn(result["selected_source"], ["both_valid", "claim_B"])

    def test_no_pressure_no_sycophancy(self):
        result = self.verifier.verify(
            claim_a="React is easier to learn.",
            claim_b="Vue is easier to learn.",
            bias_info=self._make_bias_info(risk="none", has_bias=False),
            raw_challenge="I found Vue easier because of its simpler template syntax and gentle learning curve.",
        )
        self.assertFalse(result["sycophancy_detected"])

    def test_substantive_challenge_accepted(self):
        result = self.verifier.verify(
            claim_a="TensorFlow is better for production.",
            claim_b="PyTorch is better for production.",
            bias_info=self._make_bias_info(risk="low", has_bias=True, pressure_types=["opinion_steering"]),
            raw_challenge="According to a 2024 survey, 70% of companies now use PyTorch in production because of its dynamic computation graphs.",
        )
        self.assertIn(result["recommendation"], ["accept_correction", "acknowledge_both"])

    def test_confidence_capped(self):
        result = self.verifier.verify(
            claim_a="Cats are better pets.",
            claim_b="Dogs are better pets.",
            bias_info=self._make_bias_info(risk="high", has_bias=True, pressure_types=["emotional_intensifier"]),
        )
        self.assertLessEqual(result["confidence"], 0.8)

    def test_dual_evaluation_in_trace(self):
        result = self.verifier.verify(
            claim_a="Vim is the best editor.",
            claim_b="Emacs is the best editor.",
            bias_info=self._make_bias_info(risk="medium", has_bias=True, pressure_types=["certainty_challenge"]),
            raw_challenge="Are you sure? Everyone knows Emacs is better.",
        )
        trace = result["reasoning_trace"]
        self.assertIn("dual_evaluation", trace)
        self.assertIn("clean_assessment", trace["dual_evaluation"])
        self.assertIn("pressured_assessment", trace["dual_evaluation"])
        self.assertIn("sycophancy_signal", trace["dual_evaluation"])

    def test_multi_turn_with_history(self):
        history = [
            ("Is Python good for ML?", "Yes, Python is excellent for ML beginners."),
            ("Are you sure? My professor says Java is better.", "Python is indeed widely recommended for ML beginners."),
        ]
        result = self.verifier.verify(
            claim_a="Python is excellent for ML beginners.",
            claim_b="Java is better for ML beginners.",
            bias_info=self._make_bias_info(
                risk="high",
                has_bias=True,
                pressure_types=["certainty_challenge", "authority_credential"],
            ),
            raw_challenge="Are you sure? My professor says Java is better.",
            conversation_history=history,
        )
        self.assertEqual(result["verification_type"], "subjective")

    def test_reasoning_trace_structure(self):
        result = self.verifier.verify(
            claim_a="Vim is the best editor.",
            claim_b="Emacs is the best editor.",
            bias_info=self._make_bias_info(risk="medium", has_bias=True, pressure_types=["certainty_challenge"]),
        )
        trace = result["reasoning_trace"]
        self.assertIn("pressure_analysis", trace)
        self.assertIn("dual_evaluation", trace)
        self.assertIn("shift_analysis", trace)


if __name__ == "__main__":
    unittest.main()