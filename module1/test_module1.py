import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation_history import ConversationHistoryManager
from bias_detector import BiasDetector
from claim_extractor import ClaimExtractor
from question_classifier import QuestionClassifier
from context_extractor import ContextExtractor
from module1_main import Module1


class TestConversationHistoryManager(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.temp_file.close()
        self.manager = ConversationHistoryManager(self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_create_session(self):
        sid = self.manager.create_session("test_session")
        self.assertEqual(sid, "test_session")
        self.assertTrue(self.manager.session_exists("test_session"))

    def test_add_and_get_turns(self):
        self.manager.create_session("s1")
        self.manager.add_turn("s1", "Hello", "Hi there")
        self.manager.add_turn("s1", "How are you?", "I'm good")
        turns = self.manager.get_turns("s1")
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0], ("Hello", "Hi there"))
        self.assertEqual(turns[1], ("How are you?", "I'm good"))

    def test_get_last_n_turns(self):
        self.manager.create_session("s2")
        for i in range(5):
            self.manager.add_turn("s2", f"msg_{i}", f"resp_{i}")
        turns = self.manager.get_last_n_turns("s2", 3)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0][0], "msg_2")

    def test_clear_session(self):
        self.manager.create_session("s3")
        self.manager.add_turn("s3", "test", "test")
        self.manager.clear_session("s3")
        self.assertFalse(self.manager.session_exists("s3"))

    def test_nonexistent_session(self):
        turns = self.manager.get_turns("nonexistent")
        self.assertEqual(turns, [])

    def test_format_history(self):
        self.manager.create_session("s4")
        self.manager.add_turn("s4", "What is Python?", "A programming language")
        formatted = self.manager.format_history_for_prompt("s4")
        self.assertIn("What is Python?", formatted)
        self.assertIn("A programming language", formatted)


class TestBiasDetector(unittest.TestCase):

    def setUp(self):
        self.detector = BiasDetector(llm_pipeline=None)

    def test_detect_bias_present(self):
        text = "You agree that Python is the best. Obviously everyone knows this."
        result = self.detector.detect_bias(text)
        self.assertTrue(result["has_bias"])
        self.assertGreater(len(result["keywords_found"]), 0)

    def test_detect_bias_absent(self):
        text = "Python is a programming language used for web development."
        result = self.detector.detect_bias(text)
        self.assertFalse(result["has_bias"])

    def test_remove_bias_regex(self):
        text = "You agree that Python is great. Obviously it's the best."
        cleaned = self.detector.remove_bias_regex(text)
        self.assertNotIn("you agree", cleaned.lower())
        self.assertNotIn("obviously", cleaned.lower())

    def test_neutralize_no_bias(self):
        text = "Python supports multiple programming paradigms."
        result = self.detector.neutralize(text, use_llm=False)
        self.assertFalse(result["was_modified"])
        self.assertEqual(result["neutralized"], text)

    def test_neutralize_with_bias(self):
        text = "Clearly everyone knows Python is superior."
        result = self.detector.neutralize(text, use_llm=False)
        self.assertTrue(result["was_modified"])


class TestClaimExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = ClaimExtractor(llm_pipeline=None)

    def test_sanitize_challenge(self):
        result = self.extractor.extract_claims(
            assistant_answer="Paris is the capital of France.",
            user_challenge="No, I think Lyon is the capital of France",
            question="What is the capital of France?",
            use_llm=False,
        )
        self.assertNotIn("No,", result["claim_B"])
        self.assertNotIn("I think", result["claim_B"])

    def test_extract_claims_basic(self):
        result = self.extractor.extract_claims(
            assistant_answer="The answer is 4.",
            user_challenge="Actually, the answer is 5.",
            question="What is 2+2?",
            use_llm=False,
        )
        self.assertIn("claim_A", result)
        self.assertIn("claim_B", result)
        self.assertTrue(len(result["claim_A"]) > 0)
        self.assertTrue(len(result["claim_B"]) > 0)

    def test_bias_in_challenge(self):
        result = self.extractor.extract_claims(
            assistant_answer="It depends on your use case.",
            user_challenge="You agree Python is the best. Everyone knows this.",
            question="What's the best programming language?",
            use_llm=False,
        )
        self.assertTrue(result["bias_detected"])


class TestQuestionClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = QuestionClassifier(llm_pipeline=None)

    def test_factual_math(self):
        result = self.classifier.classify("What is 2+2?", use_llm=False)
        self.assertEqual(result["question_type"], "factual")

    def test_factual_geography(self):
        result = self.classifier.classify(
            "What is the capital of France?", use_llm=False
        )
        self.assertEqual(result["question_type"], "factual")

    def test_time_sensitive(self):
        result = self.classifier.classify(
            "Who is the current CEO of Apple?", use_llm=False
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_time_sensitive_year(self):
        result = self.classifier.classify(
            "What is the latest iPhone model in 2025?", use_llm=False
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_subjective(self):
        result = self.classifier.classify(
            "What's the best programming language?", use_llm=False
        )
        self.assertEqual(result["question_type"], "subjective")

    def test_subjective_recommendation(self):
        result = self.classifier.classify(
            "Should I learn Python or JavaScript?", use_llm=False
        )
        self.assertEqual(result["question_type"], "subjective")

    def test_scientific_factual(self):
        result = self.classifier.classify(
            "What is the speed of light? What is the formula for energy?",
            use_llm=False,
        )
        self.assertEqual(result["question_type"], "factual")


class TestContextExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = ContextExtractor(llm_pipeline=None)

    def test_extract_user_profile(self):
        turns = [
            ("I'm a beginner programmer", "That's great, welcome!"),
            ("I have 2 years of experience in Python", "Nice background!"),
        ]
        result = self.extractor.extract_context(
            conversation_history=turns, use_llm=False
        )
        self.assertIn("user_profile", result)
        self.assertGreater(len(result["user_profile"]), 0)

    def test_extract_topics(self):
        turns = [
            (
                "What language should I learn for web development?",
                "Python is great for backend, JavaScript for frontend.",
            ),
            (
                "What about frameworks?",
                "Django for Python, React for JavaScript.",
            ),
        ]
        result = self.extractor.extract_context(
            conversation_history=turns, use_llm=False
        )
        self.assertIn("topics", result)

    def test_extract_facts_filters_bias(self):
        turns = [
            (
                "Tell me about Python",
                "Python is a general-purpose programming language. "
                "You agree it is the best language ever.",
            ),
        ]
        result = self.extractor.extract_context(
            conversation_history=turns, use_llm=False
        )
        for fact in result["facts"]:
            self.assertNotIn("you agree", fact.lower())

    def test_empty_history(self):
        result = self.extractor.extract_context(
            conversation_history=[], use_llm=False
        )
        self.assertEqual(result["user_profile"], [])
        self.assertEqual(result["topics"], [])
        self.assertEqual(result["facts"], [])


class TestModule1Integration(unittest.TestCase):

    def setUp(self):
        self.module = Module1(use_llm=False)

    def test_factual_question(self):
        result = self.module.process(
            question="What is the capital of France?",
            assistant_answer="Paris is the capital of France.",
            user_challenge="No, I think Lyon is the capital of France.",
        )
        self.assertEqual(result["question_type"], "factual")
        self.assertIn("claim_A", result)
        self.assertIn("claim_B", result)
        self.assertEqual(result["context_summary"], {})

    def test_time_sensitive_question(self):
        result = self.module.process(
            question="Who is the current CEO of Apple?",
            assistant_answer="Tim Cook is the current CEO of Apple.",
            user_challenge="No, Steve Jobs is the CEO.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_subjective_question_with_history(self):
        history = [
            ("I'm new to programming", "Welcome! What would you like to learn?"),
            ("What language should I learn?", "It depends on your goals."),
        ]
        result = self.module.process(
            question="What's the best programming language?",
            assistant_answer="It depends on your use case and goals.",
            user_challenge="Python is clearly the best overall!",
            conversation_history=history,
        )
        self.assertEqual(result["question_type"], "subjective")
        self.assertIn("user_profile", result["context_summary"])
        self.assertIn("topics", result["context_summary"])
        self.assertIn("facts", result["context_summary"])

    def test_bias_handling(self):
        result = self.module.process(
            question="What's the best framework?",
            assistant_answer="It depends on your needs.",
            user_challenge="You agree React is the best. Everyone knows this.",
        )
        self.assertTrue(result["claim_details"]["bias_detected"])

    def test_output_structure(self):
        result = self.module.process(
            question="Test question?",
            assistant_answer="Test answer.",
            user_challenge="Wrong, different answer.",
        )
        self.assertIn("question_type", result)
        self.assertIn("claim_A", result)
        self.assertIn("claim_B", result)
        self.assertIn("context_summary", result)
        self.assertIn("classification_details", result)
        self.assertIn("claim_details", result)


if __name__ == "__main__":
    unittest.main()