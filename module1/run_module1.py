from module1_main import Module1

print("=" * 60)
print("TEST 1: Factual Question")
print("=" * 60)
result = Module1(use_llm=False).process(
    question="What is the capital of France?",
    assistant_answer="Paris is the capital of France.",
    user_challenge="No, I think Lyon is the capital of France.",
)
print(f"Type:    {result['question_type']}")
print(f"Claim A: {result['claim_A']}")
print(f"Claim B: {result['claim_B']}")
print(f"Bias:    {result['claim_details']['bias_detected']}")
print()

print("=" * 60)
print("TEST 2: Time-Sensitive Question")
print("=" * 60)
result = Module1(use_llm=False).process(
    question="Who is the current CEO of Apple?",
    assistant_answer="Tim Cook is the current CEO of Apple.",
    user_challenge="Wrong, Steve Jobs is the CEO.",
)
print(f"Type:    {result['question_type']}")
print(f"Claim A: {result['claim_A']}")
print(f"Claim B: {result['claim_B']}")
print()

print("=" * 60)
print("TEST 3: Subjective Question with Bias + History")
print("=" * 60)
history = [
    ("I'm new to programming", "Welcome! What would you like to learn?"),
    ("What language should I learn?", "It depends on your goals."),
]
result = Module1(use_llm=False).process(
    question="What's the best programming language?",
    assistant_answer="It depends on your use case and goals.",
    user_challenge="You agree Python is the best. Everyone knows this.",
    conversation_history=history,
)
print(f"Type:    {result['question_type']}")
print(f"Claim A: {result['claim_A']}")
print(f"Claim B: {result['claim_B']}")
print(f"Bias:    {result['claim_details']['bias_detected']}")
print(f"Context: {result['context_summary']}")
print()

print("=" * 60)
print("UNIT TESTS")
print("=" * 60)
import unittest
import sys
sys.path.insert(0, "tests")
from test_module1 import (
    TestConversationHistoryManager,
    TestBiasDetector,
    TestClaimExtractor,
    TestQuestionClassifier,
    TestContextExtractor,
    TestModule1Integration,
)

loader = unittest.TestLoader()
suite = unittest.TestSuite()
for tc in [
    TestConversationHistoryManager,
    TestBiasDetector,
    TestClaimExtractor,
    TestQuestionClassifier,
    TestContextExtractor,
    TestModule1Integration,
]:
    suite.addTests(loader.loadTestsFromTestCase(tc))

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)