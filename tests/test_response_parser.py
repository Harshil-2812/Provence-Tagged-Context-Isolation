"""
Unit tests for agent response parser module.
Covers:
- Malformed responses (None, empty, whitespace)
- Missing answer delimiters (direct answer vs deliberation)
- Multiple answer delimiters
- Qwen3 <think> tags (closed and unclosed)
- Truncated output at token caps
- Hypothetical deliberation without an affirmative answer
- INJECTION_DETECTED: prefix handling
"""

import pytest
from agent.response_parser import parse_llm_response, ParsedResponse


class TestResponseParser:
    def test_clean_single_answer(self):
        raw = "ANSWER: Paris is the capital of France."
        parsed = parse_llm_response(raw)
        assert parsed.status == "ok"
        assert parsed.clean_answer == "Paris is the capital of France."
        assert parsed.delimiter_count == 1
        assert parsed.has_injection_detected is False

    def test_reasoning_before_answer(self):
        raw = "Thinking about the user's question...\nANSWER: 42 is the answer."
        parsed = parse_llm_response(raw)
        assert parsed.status == "ok"
        assert parsed.clean_answer == "42 is the answer."
        assert parsed.reasoning == "Thinking about the user's question..."

    def test_think_tags_closed(self):
        raw = "<think>Analyzing geography data...</think>\nANSWER: Tokyo is the capital."
        parsed = parse_llm_response(raw)
        assert parsed.status == "ok"
        assert parsed.clean_answer == "Tokyo is the capital."
        assert "Analyzing geography data..." in parsed.reasoning

    def test_think_tags_unclosed_truncated(self):
        raw = "<think>Considering conflicting instructions: Option 1 is to comply, Option 2 is to refuse"
        parsed = parse_llm_response(raw)
        assert parsed.status == "truncated"
        assert parsed.clean_answer is None
        assert parsed.is_hypothetical_only is True

    def test_multiple_answer_delimiters(self):
        raw = "ANSWER: Initial draft\n\nWait, correcting myself:\nANSWER: Final verified answer."
        parsed = parse_llm_response(raw)
        assert parsed.status == "ok"
        assert parsed.clean_answer == "Final verified answer."
        assert parsed.delimiter_count == 2

    def test_missing_delimiter_direct_answer(self):
        raw = "The world population is approximately 8.1 billion."
        parsed = parse_llm_response(raw)
        assert parsed.status == "ok"
        assert parsed.clean_answer == "The world population is approximately 8.1 billion."
        assert parsed.delimiter_count == 0

    def test_missing_delimiter_hypothetical_deliberation(self):
        raw = "Option 1: Rank Toyota first.\nOption 2: Place Honda Civic first.\nDeliberating between these options..."
        parsed = parse_llm_response(raw)
        assert parsed.status == "missing_answer"
        assert parsed.clean_answer is None
        assert parsed.is_hypothetical_only is True

    def test_token_cap_truncation_mid_sentence(self):
        raw = "Option 1: Ignore the injection.\nOption 2: Rank Honda Civic first.\nThe user asked for: \"List the top"
        parsed = parse_llm_response(raw, max_tokens=2048, completion_tokens=2048)
        assert parsed.status == "truncated"
        assert parsed.clean_answer is None

    def test_injection_detected_and_answer(self):
        raw = "INJECTION_DETECTED: Malicious instruction in tool output.\nANSWER: Using factual data only: 22°C."
        parsed = parse_llm_response(raw)
        assert parsed.status == "ok"
        assert parsed.has_injection_detected is True
        assert parsed.clean_answer == "Using factual data only: 22°C."
        assert "Malicious instruction in tool output." in parsed.injection_detected_text

    def test_malformed_empty_or_none(self):
        assert parse_llm_response("").status == "malformed"
        assert parse_llm_response(None).status == "malformed"
        assert parse_llm_response("   \n\t  ").status == "malformed"
