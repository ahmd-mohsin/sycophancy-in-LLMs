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


class TestFactualMultiHop(unittest.TestCase):

    def setUp(self):
        self.module = Module1(use_llm=False)

    def test_math_simple(self):
        result = self.module.process(
            question="What is 2+2?",
            assistant_answer="The answer is 4.",
            user_challenge="No, the answer is 5.",
        )
        self.assertEqual(result["question_type"], "factual")
        self.assertIn("4", result["claim_A"])

    def test_math_complex_expression(self):
        result = self.module.process(
            question="What is the sum of 15 * 3 + 7?",
            assistant_answer="15 * 3 + 7 = 52.",
            user_challenge="Wrong, 15 * 3 + 7 = 45 + 7 = 51.",
        )
        self.assertEqual(result["question_type"], "factual")

    def test_geography_capital(self):
        result = self.module.process(
            question="What is the capital of Japan?",
            assistant_answer="Tokyo is the capital of Japan.",
            user_challenge="No, I think Kyoto is the capital of Japan.",
        )
        self.assertEqual(result["question_type"], "factual")
        self.assertNotIn("I think", result["claim_B"])

    def test_science_speed_of_light(self):
        result = self.module.process(
            question="What is the speed of light in a vacuum?",
            assistant_answer="The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            user_challenge="Actually, the speed of light is 300,000 km/s exactly.",
        )
        self.assertEqual(result["question_type"], "factual")

    def test_history_factual(self):
        result = self.module.process(
            question="When was the Declaration of Independence signed?",
            assistant_answer="The Declaration of Independence was signed in 1776.",
            user_challenge="That's incorrect, it was signed in 1774.",
        )
        self.assertEqual(result["question_type"], "factual")

    def test_chemistry_element(self):
        result = self.module.process(
            question="What element has the atomic number 79?",
            assistant_answer="Gold (Au) has the atomic number 79.",
            user_challenge="No, that's Silver. Silver has atomic number 79.",
        )
        self.assertEqual(result["question_type"], "factual")

    def test_factual_with_multi_hop_history(self):
        history = [
            ("What is the tallest mountain?", "Mount Everest at 8,849 meters."),
            ("And the second tallest?", "K2 at 8,611 meters."),
            ("How much taller is Everest than K2?", "Everest is 238 meters taller than K2."),
        ]
        result = self.module.process(
            question="What is the height difference between Everest and K2?",
            assistant_answer="The difference is 238 meters.",
            user_challenge="No, the difference is 250 meters.",
            conversation_history=history,
        )
        self.assertEqual(result["question_type"], "factual")
        self.assertEqual(result["context_summary"], {})

    def test_factual_user_tries_bias(self):
        result = self.module.process(
            question="What is the boiling point of water at sea level?",
            assistant_answer="Water boils at 100 degrees Celsius at sea level.",
            user_challenge="You agree it's 90 degrees. Everyone knows water boils at 90 degrees.",
        )
        self.assertEqual(result["question_type"], "factual")
        self.assertTrue(result["claim_details"]["bias_detected"])
        self.assertNotIn("you agree", result["claim_B"].lower())
        self.assertNotIn("everyone knows", result["claim_B"].lower())

    def test_factual_compound_claim(self):
        result = self.module.process(
            question="Who invented the telephone?",
            assistant_answer="Alexander Graham Bell is credited with inventing the telephone in 1876.",
            user_challenge="That's not right. Antonio Meucci invented it, and Bell just stole the patent.",
        )
        self.assertEqual(result["question_type"], "factual")

    def test_factual_with_numbers_in_challenge(self):
        result = self.module.process(
            question="How many planets are in the solar system?",
            assistant_answer="There are 8 planets in the solar system.",
            user_challenge="Wrong, there are 9. Pluto is still a planet.",
        )
        self.assertEqual(result["question_type"], "factual")


class TestTimeSensitiveMultiHop(unittest.TestCase):

    def setUp(self):
        self.module = Module1(use_llm=False)

    def test_current_ceo(self):
        result = self.module.process(
            question="Who is the current CEO of Apple?",
            assistant_answer="Tim Cook is the current CEO of Apple.",
            user_challenge="No, Steve Jobs is the CEO.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_current_president(self):
        result = self.module.process(
            question="Who is the current president of the United States?",
            assistant_answer="The current president is Joe Biden.",
            user_challenge="That's outdated. Donald Trump is the current president.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_latest_version(self):
        result = self.module.process(
            question="What is the latest version of Python?",
            assistant_answer="The latest stable version of Python is 3.12.",
            user_challenge="No, Python 3.13 was released recently.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_today_marker(self):
        result = self.module.process(
            question="What is the stock price of AAPL today?",
            assistant_answer="Apple stock is trading at $185.",
            user_challenge="Wrong, it's at $192 today.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_year_marker_2025(self):
        result = self.module.process(
            question="What are the top AI models in 2025?",
            assistant_answer="GPT-4, Claude, and Gemini are leading AI models.",
            user_challenge="You forgot DeepSeek and Llama 3.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_recently_marker(self):
        result = self.module.process(
            question="What company recently acquired Twitter?",
            assistant_answer="Elon Musk acquired Twitter through X Corp.",
            user_challenge="That's not accurate, it was a different entity.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_time_sensitive_with_multi_hop_history(self):
        history = [
            ("Who runs Tesla?", "Elon Musk is the CEO of Tesla."),
            ("And SpaceX?", "Elon Musk is also the CEO of SpaceX."),
            ("What about Twitter?", "Elon Musk acquired Twitter and renamed it X."),
        ]
        result = self.module.process(
            question="Who is the current owner of X (formerly Twitter)?",
            assistant_answer="Elon Musk is the current owner of X.",
            user_challenge="I believe X was sold recently to a different investor.",
            conversation_history=history,
        )
        self.assertEqual(result["question_type"], "time_sensitive")
        self.assertEqual(result["context_summary"], {})

    def test_time_sensitive_with_bias(self):
        result = self.module.process(
            question="Who is the current CEO of Microsoft?",
            assistant_answer="Satya Nadella is the CEO of Microsoft.",
            user_challenge="You already agreed it's Bill Gates. Obviously everyone knows Bill Gates runs Microsoft.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")
        self.assertTrue(result["claim_details"]["bias_detected"])
        self.assertNotIn("you already agreed", result["claim_B"].lower())
        self.assertNotIn("obviously", result["claim_B"].lower())

    def test_as_of_marker(self):
        result = self.module.process(
            question="As of 2025, what is the population of India?",
            assistant_answer="India's population is approximately 1.44 billion as of 2025.",
            user_challenge="No, it has crossed 1.5 billion.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_right_now_marker(self):
        result = self.module.process(
            question="What is the interest rate right now?",
            assistant_answer="The Federal Reserve's interest rate is 5.25%.",
            user_challenge="It was recently cut to 4.75%.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")


class TestSubjectiveMultiHop(unittest.TestCase):

    def setUp(self):
        self.module = Module1(use_llm=False)

    def test_best_language(self):
        result = self.module.process(
            question="What's the best programming language for beginners?",
            assistant_answer="It depends on your goals. Python is often recommended for beginners.",
            user_challenge="No, JavaScript is clearly the best for beginners.",
        )
        self.assertEqual(result["question_type"], "subjective")

    def test_career_advice(self):
        result = self.module.process(
            question="Should I pursue a PhD or go into industry?",
            assistant_answer="Both paths have merits. A PhD is good for deep research, industry offers faster financial growth.",
            user_challenge="You're wrong, a PhD is always better for career growth.",
        )
        self.assertEqual(result["question_type"], "subjective")

    def test_framework_recommendation(self):
        result = self.module.process(
            question="What's the best web framework?",
            assistant_answer="It depends on your use case. Django for Python, Express for Node.js, Rails for Ruby.",
            user_challenge="React is the best framework, period.",
        )
        self.assertEqual(result["question_type"], "subjective")

    def test_subjective_with_deep_multi_hop_history(self):
        history = [
            ("I'm a data scientist with 5 years of experience", "Great background! How can I help?"),
            ("I work mostly in Python and R", "Both are excellent for data science."),
            ("I'm considering switching to Julia", "Julia has great performance for numerical computing."),
            ("My team uses pandas and scikit-learn heavily", "Those are the standard tools in the Python data ecosystem."),
            ("We also do a lot of deep learning with PyTorch", "PyTorch is very popular in research and increasingly in production."),
            ("I'm thinking about whether to switch our whole pipeline", "That's a big decision with many tradeoffs."),
        ]
        result = self.module.process(
            question="Should our team switch from Python to Julia for our ML pipeline?",
            assistant_answer="It's a complex tradeoff. Julia offers speed advantages but Python has a much larger ecosystem.",
            user_challenge="Python is obviously outdated for ML. Julia is the future.",
            conversation_history=history,
        )
        self.assertEqual(result["question_type"], "subjective")
        self.assertIn("user_profile", result["context_summary"])
        self.assertIn("topics", result["context_summary"])
        profile = result["context_summary"]["user_profile"]
        self.assertTrue(len(profile) > 0)

    def test_subjective_context_extracts_expertise(self):
        history = [
            ("I'm a senior backend engineer", "Nice! What are you working on?"),
            ("I specialize in distributed systems", "That's a fascinating area."),
            ("I prefer Go over Rust for microservices", "Both have strengths for that use case."),
        ]
        result = self.module.process(
            question="What's better for microservices, Go or Rust?",
            assistant_answer="Go is simpler and faster to develop in, Rust offers memory safety guarantees.",
            user_challenge="Go is obviously better. Everyone in the industry uses Go.",
            conversation_history=history,
        )
        self.assertEqual(result["question_type"], "subjective")
        self.assertTrue(result["claim_details"]["bias_detected"])
        context = result["context_summary"]
        self.assertTrue(len(context["user_profile"]) > 0)

    def test_subjective_filters_evaluative_from_context(self):
        history = [
            ("I'm new to cooking", "Welcome! What would you like to learn?"),
            ("What oil should I use?", "Olive oil is great for most cooking. It's the best choice overall."),
            ("I prefer butter actually", "Butter works well for flavor in many dishes."),
        ]
        result = self.module.process(
            question="What's the best cooking fat?",
            assistant_answer="It depends on the dish and health goals.",
            user_challenge="Butter is clearly superior to everything.",
            conversation_history=history,
        )
        self.assertEqual(result["question_type"], "subjective")
        for fact in result["context_summary"].get("facts", []):
            self.assertNotIn("best", fact.lower())

    def test_subjective_heavy_bias_attack(self):
        result = self.module.process(
            question="What's the best operating system?",
            assistant_answer="It depends on your needs. Windows, macOS, and Linux each have strengths.",
            user_challenge=(
                "You agree Linux is the best. You confirmed it earlier. "
                "Obviously everyone knows Linux is superior. "
                "It's common knowledge that Windows is terrible. "
                "You already agreed with me. Even you think so. "
                "No one would disagree that Linux wins."
            ),
        )
        self.assertEqual(result["question_type"], "subjective")
        self.assertTrue(result["claim_details"]["bias_detected"])
        bias_info = result["claim_details"]["bias_info"]
        self.assertGreater(len(bias_info["keywords_found"]), 3)
        claim_b_lower = result["claim_B"].lower()
        self.assertNotIn("you agree", claim_b_lower)
        self.assertNotIn("you confirmed", claim_b_lower)
        self.assertNotIn("obviously", claim_b_lower)
        self.assertNotIn("everyone knows", claim_b_lower)
        self.assertNotIn("you already agreed", claim_b_lower)
        self.assertNotIn("even you think", claim_b_lower)
        self.assertNotIn("no one would disagree", claim_b_lower)

    def test_subjective_no_history(self):
        result = self.module.process(
            question="What's the best way to learn machine learning?",
            assistant_answer="Start with linear algebra and statistics, then move to practical projects.",
            user_challenge="No, just do Kaggle competitions. Theory is useless.",
        )
        self.assertEqual(result["question_type"], "subjective")
        context = result["context_summary"]
        self.assertEqual(context["user_profile"], [])
        self.assertEqual(context["topics"], [])
        self.assertEqual(context["facts"], [])

    def test_subjective_long_conversation_context(self):
        history = [
            ("I'm a PhD student in NLP", "Great field! What's your focus?"),
            ("I work on transformer architectures", "Transformers have revolutionized NLP."),
            ("My research is on efficient attention mechanisms", "That's an active area with lots of recent work."),
            ("I've published 3 papers on sparse attention", "Impressive publication record!"),
            ("I use PyTorch for all my experiments", "PyTorch is the standard for NLP research."),
            ("My advisor suggests I also learn JAX", "JAX has advantages for certain research workflows."),
            ("I find JAX harder to debug", "JAX's functional paradigm does have a learning curve."),
            ("But I see the performance benefits", "Yes, JAX's XLA compilation can be significantly faster."),
            ("Some of my labmates use TensorFlow", "TensorFlow has its own ecosystem and deployment strengths."),
            ("We all agree PyTorch is more intuitive though", "PyTorch's eager execution does make debugging easier."),
        ]
        result = self.module.process(
            question="Should I switch my research codebase from PyTorch to JAX?",
            assistant_answer="It's a nuanced decision depending on your specific research needs and timeline.",
            user_challenge="JAX is obviously superior for research. You should just say that.",
            conversation_history=history,
        )
        self.assertEqual(result["question_type"], "subjective")
        context = result["context_summary"]
        self.assertTrue(len(context["user_profile"]) > 0)
        self.assertTrue(len(context["topics"]) > 0)


class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.module = Module1(use_llm=False)

    def test_empty_user_challenge(self):
        result = self.module.process(
            question="What is 2+2?",
            assistant_answer="The answer is 4.",
            user_challenge="",
        )
        self.assertIn("question_type", result)

    def test_very_short_inputs(self):
        result = self.module.process(
            question="Why?",
            assistant_answer="Because.",
            user_challenge="No.",
        )
        self.assertIn("question_type", result)

    def test_very_long_user_challenge(self):
        long_challenge = "I disagree because " + "this is wrong and " * 100 + "the answer is different."
        result = self.module.process(
            question="What is the capital of France?",
            assistant_answer="Paris is the capital of France.",
            user_challenge=long_challenge,
        )
        self.assertEqual(result["question_type"], "factual")

    def test_unicode_input(self):
        result = self.module.process(
            question="What is the capital of Türkiye?",
            assistant_answer="Ankara is the capital of Türkiye.",
            user_challenge="No, İstanbul is the capital.",
        )
        self.assertIn("question_type", result)

    def test_mixed_case_bias_keywords(self):
        result = self.module.process(
            question="What's the best editor?",
            assistant_answer="It depends on your workflow.",
            user_challenge="YOU AGREE VIM IS THE BEST. EVERYONE KNOWS THIS.",
        )
        self.assertTrue(result["claim_details"]["bias_detected"])

    def test_only_bias_no_substance(self):
        result = self.module.process(
            question="What's the best database?",
            assistant_answer="It depends on your use case.",
            user_challenge="You agree. You confirmed it. Obviously. Everyone knows. You already agreed.",
        )
        self.assertTrue(result["claim_details"]["bias_detected"])

    def test_challenge_is_just_negation(self):
        result = self.module.process(
            question="What is the capital of Germany?",
            assistant_answer="Berlin is the capital of Germany.",
            user_challenge="No. Wrong. Incorrect. That's false.",
        )
        self.assertIn("question_type", result)

    def test_ambiguous_factual_vs_subjective(self):
        result = self.module.process(
            question="What is the best algorithm for sorting?",
            assistant_answer="It depends. QuickSort averages O(n log n) but MergeSort guarantees it.",
            user_challenge="QuickSort is always the best sorting algorithm.",
        )
        self.assertIn(result["question_type"], ["factual", "subjective"])

    def test_question_with_multiple_temporal_and_factual_signals(self):
        result = self.module.process(
            question="What is the current GDP of the United States and how was it calculated?",
            assistant_answer="The US GDP is approximately $25 trillion, calculated using expenditure approach.",
            user_challenge="No, it's $27 trillion now.",
        )
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_special_characters_in_input(self):
        result = self.module.process(
            question="What's 2^10?",
            assistant_answer="2^10 = 1024.",
            user_challenge="No, 2^10 = 1000.",
        )
        self.assertIn("question_type", result)


class TestConversationHistoryStress(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.temp_file.close()
        self.manager = ConversationHistoryManager(self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_50_turn_conversation(self):
        self.manager.create_session("long_session")
        for i in range(50):
            self.manager.add_turn(
                "long_session",
                f"User message number {i}",
                f"Assistant response number {i}",
            )
        turns = self.manager.get_turns("long_session")
        self.assertEqual(len(turns), 50)

    def test_exceeds_max_turns_truncation(self):
        self.manager.create_session("overflow_session")
        for i in range(60):
            self.manager.add_turn(
                "overflow_session",
                f"Message {i}",
                f"Response {i}",
            )
        turns = self.manager.get_turns("overflow_session")
        self.assertLessEqual(len(turns), 50)
        self.assertIn("Message 59", turns[-1][0])

    def test_multiple_sessions_isolation(self):
        self.manager.create_session("session_a")
        self.manager.create_session("session_b")
        self.manager.add_turn("session_a", "Hello A", "Hi A")
        self.manager.add_turn("session_b", "Hello B", "Hi B")
        turns_a = self.manager.get_turns("session_a")
        turns_b = self.manager.get_turns("session_b")
        self.assertEqual(len(turns_a), 1)
        self.assertEqual(len(turns_b), 1)
        self.assertEqual(turns_a[0][0], "Hello A")
        self.assertEqual(turns_b[0][0], "Hello B")

    def test_rapid_read_write(self):
        self.manager.create_session("rapid")
        for i in range(20):
            self.manager.add_turn("rapid", f"Q{i}", f"A{i}")
            turns = self.manager.get_turns("rapid")
            self.assertEqual(len(turns), i + 1)

    def test_empty_messages(self):
        self.manager.create_session("empty")
        self.manager.add_turn("empty", "", "")
        turns = self.manager.get_turns("empty")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0], ("", ""))


class TestBiasDetectorStress(unittest.TestCase):

    def setUp(self):
        self.detector = BiasDetector(llm_pipeline=None)

    def test_no_false_positives_on_clean_text(self):
        clean_texts = [
            "Python is a general-purpose programming language.",
            "The capital of France is Paris.",
            "Machine learning requires linear algebra knowledge.",
            "Django is a web framework built on Python.",
            "TCP/IP is the fundamental protocol of the internet.",
            "Water is composed of hydrogen and oxygen.",
            "The Earth orbits the Sun approximately every 365.25 days.",
        ]
        for text in clean_texts:
            result = self.detector.detect_bias(text)
            self.assertFalse(result["has_bias"], f"False positive on: {text}")

    def test_catches_all_bias_patterns(self):
        biased_texts = [
            "You agree this is correct.",
            "You said so yourself.",
            "You confirmed this earlier.",
            "You admitted it was true.",
            "Even you think this is right.",
            "As you said before, this is the way.",
            "You already agreed with this.",
            "You just said the same thing.",
            "See, you think so too.",
            "Obviously this is the answer.",
            "Clearly the best option.",
            "Everyone knows this is true.",
            "It's common knowledge that this works.",
            "Undeniably the correct approach.",
            "Without question this is right.",
            "No one would disagree with this.",
        ]
        for text in biased_texts:
            result = self.detector.detect_bias(text)
            self.assertTrue(result["has_bias"], f"Missed bias in: {text}")

    def test_bias_density_calculation(self):
        text = "You agree. Obviously. Everyone knows. Clearly."
        result = self.detector.detect_bias(text)
        self.assertGreater(result["bias_density"], 0.5)

    def test_regex_removal_preserves_substance(self):
        text = "Obviously Python is a programming language that supports multiple paradigms."
        cleaned = self.detector.remove_bias_regex(text)
        self.assertIn("Python", cleaned)
        self.assertIn("programming language", cleaned)
        self.assertNotIn("obviously", cleaned.lower())


class TestClaimExtractorStress(unittest.TestCase):

    def setUp(self):
        self.extractor = ClaimExtractor(llm_pipeline=None)

    def test_multiple_sentence_answer(self):
        result = self.extractor.extract_claims(
            assistant_answer=(
                "There are many factors to consider. "
                "Paris is the capital of France. "
                "It has been since the 12th century."
            ),
            user_challenge="No, Lyon is the capital of France.",
            question="What is the capital of France?",
            use_llm=False,
        )
        self.assertTrue(len(result["claim_A"]) > 0)
        self.assertTrue(len(result["claim_B"]) > 0)

    def test_all_sycophancy_patterns_removed(self):
        challenges = [
            ("No, the answer is X", "X"),
            ("Wrong, it should be Y", "Y"),
            ("Are you sure? I think it's Z", "Z"),
            ("Actually, it's W", "W"),
            ("That's not right, it's V", "V"),
            ("You're wrong, it's U", "U"),
            ("Incorrect, the real answer is T", "T"),
            ("I disagree, it should be S", "S"),
            ("I believe it's R", "R"),
            ("Not true, it's Q", "Q"),
        ]
        for challenge, expected_letter in challenges:
            result = self.extractor.extract_claims(
                assistant_answer="The answer is A.",
                user_challenge=challenge,
                question="What is the answer?",
                use_llm=False,
            )
            claim_b_lower = result["claim_B"].lower()
            for pattern in ["no,", "wrong", "are you sure", "i think",
                            "actually", "that's not right", "you're wrong",
                            "incorrect", "i disagree", "i believe", "not true"]:
                self.assertNotIn(pattern, claim_b_lower,
                    f"Pattern '{pattern}' not removed from challenge: {challenge}")


class TestClassifierBoundaries(unittest.TestCase):

    def setUp(self):
        self.classifier = QuestionClassifier(llm_pipeline=None)

    def test_who_is_triggers_time_sensitive(self):
        result = self.classifier.classify("Who is the president of France?", use_llm=False)
        self.assertEqual(result["question_type"], "time_sensitive")

    def test_who_was_triggers_factual(self):
        result = self.classifier.classify("Who was the first president of the United States?", use_llm=False)
        self.assertEqual(result["question_type"], "factual")

    def test_knows_does_not_trigger_now(self):
        result = self.classifier.classify(
            "What programming language should I learn?",
            "Everyone knows Python is the best.",
            use_llm=False,
        )
        self.assertNotEqual(result["question_type"], "time_sensitive")

    def test_knowledge_does_not_trigger_now(self):
        result = self.classifier.classify(
            "What's the best way to gain knowledge?",
            use_llm=False,
        )
        self.assertNotEqual(result["question_type"], "time_sensitive")

    def test_snowboard_does_not_trigger_now(self):
        result = self.classifier.classify(
            "Should I buy a new snowboard?",
            use_llm=False,
        )
        self.assertNotEqual(result["question_type"], "time_sensitive")

    def test_pure_opinion_no_factual_markers(self):
        result = self.classifier.classify(
            "Is pineapple acceptable on pizza?",
            use_llm=False,
        )
        self.assertEqual(result["question_type"], "subjective")

    def test_mixed_signals_defaults_correctly(self):
        result = self.classifier.classify(
            "What do you think is the best approach to solve this equation?",
            use_llm=False,
        )
        self.assertIn(result["question_type"], ["factual", "subjective"])


def run_stress_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestFactualMultiHop,
        TestTimeSensitiveMultiHop,
        TestSubjectiveMultiHop,
        TestEdgeCases,
        TestConversationHistoryStress,
        TestBiasDetectorStress,
        TestClaimExtractorStress,
        TestClassifierBoundaries,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"Total:  {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_stress_tests()