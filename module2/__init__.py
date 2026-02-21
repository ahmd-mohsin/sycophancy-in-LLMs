"""
Module 2 — Verification Framework
===================================
Takes Module 1 output (core claim, bias info, question type, sycophancy score)
and routes to the appropriate verifier.

Module 2 NEVER re-runs detection.  It only verifies and generates unbiased positions.
"""