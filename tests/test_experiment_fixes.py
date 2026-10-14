"""
Tests for pilot-readiness fixes:
  FIX 1 – Complete raw LLM response is preserved in result dicts.
  FIX 2 – API error trials are marked 'error' and excluded from all metrics.

No real LLM API calls are made. All tests use either:
  - The mock LLM provider, or
  - Manually constructed result dicts.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

from agent.agent import Agent, LLMClient, AgentResponse
from attacks.indirect_injection import load_attack_cases
from evaluation.experiment_runner import (
    run_attack_experiment,
    run_task_experiment,
    load_task_cases,
)
from evaluation.metrics import (
    compute_attack_success_rate,
    compute_attack_blocking_rate,
    compute_task_success_rate,
    compute_false_blocking_rate,
    compute_all_metrics,
    compute_error_counts,
    _valid_attack_results,
    _valid_task_results,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_agent(defense_mode: str = "no_defense") -> Agent:
    llm = LLMClient(provider="mock")
    return Agent(llm_client=llm, defense_mode=defense_mode)


def _first_attack_case():
    cases = load_attack_cases()
    assert cases, "attack_cases.json must not be empty"
    return cases[0]


def _first_task_case() -> dict:
    cases = load_task_cases()
    assert cases, "task_cases.json must not be empty"
    return cases[0]


def _make_error_response(error_msg: str = "Simulated API timeout") -> AgentResponse:
    """Construct an AgentResponse that mirrors what Agent.run() returns on API failure."""
    return AgentResponse(
        raw_response="",
        proposed_action="",
        validation_result=None,
        final_decision="ERROR",
        context_mode="raw",
        defense_mode="no_defense",
        latency_ms=0.0,
        token_usage={},
        context_summary={},
        error=error_msg,
    )


def _make_mixed_results(n_valid_attacks=3, n_error_attacks=1,
                        n_valid_tasks=3, n_error_tasks=1,
                        attack_success=True):
    """Build a mixed result list with valid and error records."""
    results = []
    for _ in range(n_valid_attacks):
        results.append({
            "is_attack": True,
            "status": "ok",
            "llm_compliance": attack_success,
            "end_to_end_success": attack_success,
            "attack_success": attack_success,
            "task_success": not attack_success,
            "final_decision": "EXECUTED" if attack_success else "BLOCKED",
            "false_block": False,
            "defense": "test",
            "latency_ms": 100.0,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
    for _ in range(n_error_attacks):
        results.append({
            "is_attack": True,
            "status": "error",
            "llm_compliance": None,
            "end_to_end_success": None,
            "attack_success": None,
            "task_success": None,
            "final_decision": "ERROR",
            "false_block": None,
            "defense": "test",
            "latency_ms": 0.0,
            "token_usage": {},
        })
    for _ in range(n_valid_tasks):
        results.append({
            "is_attack": False,
            "status": "ok",
            "task_success": True,
            "false_block": False,
            "final_decision": "EXECUTED",
            "llm_compliance": False,
            "defense": "test",
            "latency_ms": 80.0,
            "token_usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        })
    for _ in range(n_error_tasks):
        results.append({
            "is_attack": False,
            "status": "error",
            "task_success": None,
            "false_block": None,
            "final_decision": "ERROR",
            "llm_compliance": None,
            "defense": "test",
            "latency_ms": 0.0,
            "token_usage": {},
        })
    return results


# ─── FIX 1: Complete raw_response preservation ────────────────────────────────

class TestRawResponsePreservation:
    """FIX 1: The result dict must store the full raw LLM response, not a truncation."""

    def test_attack_result_has_raw_response_field(self):
        """Attack result must contain a 'raw_response' key."""
        agent = _mock_agent("no_defense")
        result = run_attack_experiment(agent, _first_attack_case(), "no_defense")
        assert "raw_response" in result, "raw_response field is missing from attack result"

    def test_task_result_has_raw_response_field(self):
        """Task result must contain a 'raw_response' key."""
        agent = _mock_agent("no_defense")
        result = run_task_experiment(agent, _first_task_case(), "no_defense")
        assert "raw_response" in result, "raw_response field is missing from task result"

    def test_raw_response_is_string(self):
        """raw_response must be a string, never None."""
        agent = _mock_agent("no_defense")
        result = run_task_experiment(agent, _first_task_case(), "no_defense")
        assert isinstance(result["raw_response"], str)

    def test_500_char_response_is_not_truncated(self):
        """
        A 500-character model response must be stored in full in raw_response.
        The snippet field may cap at 200 chars; raw_response must not.
        """
        long_response = "ANSWER: " + ("X" * 500)
        agent = _mock_agent("no_defense")
        with patch.object(agent.llm, "complete", return_value=(long_response, {})):
            result = run_task_experiment(agent, _first_task_case(), "no_defense")

        assert result["raw_response"] == long_response, (
            "raw_response was truncated – it must store the complete model output"
        )
        assert len(result["raw_response"]) == len(long_response)

    def test_raw_response_snippet_is_at_most_200_chars(self):
        """raw_response_snippet must still be capped at 200 characters."""
        long_response = "ANSWER: " + ("Y" * 500)
        agent = _mock_agent("no_defense")
        with patch.object(agent.llm, "complete", return_value=(long_response, {})):
            result = run_task_experiment(agent, _first_task_case(), "no_defense")

        assert len(result.get("raw_response_snippet", "")) <= 200, (
            "raw_response_snippet should be at most 200 characters"
        )

    def test_raw_response_is_json_serialisable(self):
        """A result dict including raw_response must be JSON-serialisable."""
        long_response = "ANSWER: unicode safe " + "Z" * 300
        agent = _mock_agent("full_defense")
        with patch.object(agent.llm, "complete", return_value=(long_response, {})):
            result = run_task_experiment(agent, _first_task_case(), "full_defense")
        serialised = json.dumps(result, default=str)
        assert "raw_response" in json.loads(serialised)

    def test_snippet_is_prefix_of_full_response_on_ok_trials(self):
        """raw_response_snippet must be a prefix of raw_response on successful trials."""
        agent = _mock_agent("no_defense")
        result = run_task_experiment(agent, _first_task_case(), "no_defense")
        if result.get("status") == "ok":
            full = result["raw_response"]
            snippet = result.get("raw_response_snippet", "")
            assert full.startswith(snippet), (
                "raw_response_snippet is not a prefix of raw_response"
            )


# ─── FIX 2: API error isolation ───────────────────────────────────────────────

class TestAPIErrorIsolation:
    """FIX 2: API error trials must be quarantined and excluded from all metrics."""

    # ── Detection ─────────────────────────────────────────────────────────────

    def test_attack_error_is_marked_with_status_error(self):
        """When Agent.run() returns ERROR, the attack result must have status='error'."""
        agent = _mock_agent("no_defense")
        with patch.object(agent, "run", return_value=_make_error_response()):
            result = run_attack_experiment(agent, _first_attack_case(), "no_defense")
        assert result["status"] == "error"

    def test_task_error_is_marked_with_status_error(self):
        """When Agent.run() returns ERROR, the task result must have status='error'."""
        agent = _mock_agent("no_defense")
        with patch.object(agent, "run", return_value=_make_error_response()):
            result = run_task_experiment(agent, _first_task_case(), "no_defense")
        assert result["status"] == "error"

    def test_error_result_preserves_error_message(self):
        """Error description must be stored in the result."""
        agent = _mock_agent("no_defense")
        with patch.object(agent, "run", return_value=_make_error_response("Rate limit exceeded")):
            result = run_attack_experiment(agent, _first_attack_case(), "no_defense")
        assert "error" in result
        assert result["error"]  # Non-empty string

    # ── No false success classification ───────────────────────────────────────

    def test_api_error_does_not_become_llm_compliance_false(self):
        """
        llm_compliance must be None for error trials – never the boolean False
        that would make a failed API call look like a successfully defended attack.
        """
        agent = _mock_agent("no_defense")
        with patch.object(agent, "run", return_value=_make_error_response()):
            result = run_attack_experiment(agent, _first_attack_case(), "no_defense")
        assert result["llm_compliance"] is None, (
            "llm_compliance should be None for error trials, not False"
        )

    def test_api_error_does_not_become_task_success_true(self):
        """
        task_success must be None for error task trials – never True,
        which would falsely inflate TSR.
        """
        agent = _mock_agent("no_defense")
        with patch.object(agent, "run", return_value=_make_error_response()):
            result = run_task_experiment(agent, _first_task_case(), "no_defense")
        assert result["task_success"] is None, (
            "task_success should be None for error trials, not True"
        )

    def test_api_error_attack_final_decision_is_ERROR(self):
        """final_decision must remain 'ERROR', not 'EXECUTED' or 'BLOCKED'."""
        agent = _mock_agent("no_defense")
        with patch.object(agent, "run", return_value=_make_error_response()):
            result = run_attack_experiment(agent, _first_attack_case(), "no_defense")
        assert result["final_decision"] == "ERROR"

    def test_api_error_task_final_decision_is_ERROR(self):
        """Task error final_decision must remain 'ERROR'."""
        agent = _mock_agent("no_defense")
        with patch.object(agent, "run", return_value=_make_error_response()):
            result = run_task_experiment(agent, _first_task_case(), "no_defense")
        assert result["final_decision"] == "ERROR"

    def test_api_error_is_not_classified_as_false_block(self):
        """false_block must be None for error trials, not True/False."""
        agent = _mock_agent("no_defense")
        with patch.object(agent, "run", return_value=_make_error_response()):
            result = run_task_experiment(agent, _first_task_case(), "no_defense")
        assert result.get("false_block") is None

    # ── Metric exclusion ──────────────────────────────────────────────────────

    def test_error_trials_excluded_from_attack_denominator(self):
        """
        3 valid + 1 error = 4 total attack trials.
        With all 3 valid attacks succeeding: ASR denominator must be 3, so ASR = 1.0.
        """
        results = _make_mixed_results(
            n_valid_attacks=3, n_error_attacks=1, attack_success=True,
            n_valid_tasks=0, n_error_tasks=0,
        )
        asr = compute_attack_success_rate(results)
        assert asr == pytest.approx(1.0), (
            f"ASR should be 1.0 (3/3 valid) but got {asr} – error trial polluted denominator"
        )

    def test_error_trials_excluded_from_task_denominator(self):
        """
        3 valid + 1 error = 4 total task trials.
        With all 3 valid tasks succeeding: TSR denominator must be 3, so TSR = 1.0.
        """
        results = _make_mixed_results(
            n_valid_attacks=0, n_error_attacks=0,
            n_valid_tasks=3, n_error_tasks=1,
        )
        tsr = compute_task_success_rate(results)
        assert tsr == pytest.approx(1.0), (
            f"TSR should be 1.0 (3/3 valid) but got {tsr} – error trial polluted denominator"
        )

    def test_error_trials_excluded_from_abr_denominator(self):
        """ABR denominator must be valid attacks only."""
        results = _make_mixed_results(
            n_valid_attacks=4, n_error_attacks=2, attack_success=False,
            n_valid_tasks=0, n_error_tasks=0,
        )
        # Force all valid attacks to final_decision=BLOCKED
        for r in results:
            if r.get("is_attack") and r.get("status") == "ok":
                r["final_decision"] = "BLOCKED"
        abr = compute_attack_blocking_rate(results)
        assert abr == pytest.approx(1.0), (
            f"ABR should be 1.0 (4/4 valid blocked) but got {abr}"
        )

    def test_error_count_reported_in_all_metrics(self):
        """compute_all_metrics must expose error_trials, valid_trials, total_attempted."""
        results = _make_mixed_results(
            n_valid_attacks=2, n_error_attacks=2,
            n_valid_tasks=2, n_error_tasks=2,
        )
        m = compute_all_metrics(results, "test")
        assert "error_trials" in m
        assert "valid_trials" in m
        assert "total_attempted" in m
        assert m["error_trials"] == 4
        assert m["total_attempted"] == 8
        assert m["valid_trials"] == 4

    def test_all_errors_no_zero_division(self):
        """
        When ALL trials are errors, every rate metric must return 0.0 safely
        without raising ZeroDivisionError.
        """
        results = _make_mixed_results(
            n_valid_attacks=0, n_error_attacks=3,
            n_valid_tasks=0, n_error_tasks=3,
        )
        assert compute_attack_success_rate(results) == 0.0
        assert compute_task_success_rate(results) == 0.0
        assert compute_attack_blocking_rate(results) == 0.0
        assert compute_false_blocking_rate(results) == 0.0

    def test_compute_error_counts_helper(self):
        """compute_error_counts returns correct breakdowns."""
        results = _make_mixed_results(
            n_valid_attacks=3, n_error_attacks=1,
            n_valid_tasks=4, n_error_tasks=2,
        )
        counts = compute_error_counts(results)
        assert counts["total_attempted"] == 10
        assert counts["error_trials"] == 3
        assert counts["valid_trials"] == 7
        assert counts["attack_error_trials"] == 1
        assert counts["task_error_trials"] == 2

    def test_valid_attack_filter_excludes_none_compliance(self):
        """_valid_attack_results must exclude records where llm_compliance is None."""
        results = [
            {"is_attack": True, "status": "ok",    "llm_compliance": True,  "task_success": False},
            {"is_attack": True, "status": "error",  "llm_compliance": None,  "task_success": None},
            {"is_attack": True, "status": "ok",    "llm_compliance": False, "task_success": True},
        ]
        valid = _valid_attack_results(results)
        assert len(valid) == 2
        assert all(r["llm_compliance"] is not None for r in valid)

    def test_valid_task_filter_excludes_none_task_success(self):
        """_valid_task_results must exclude records where task_success is None."""
        results = [
            {"is_attack": False, "status": "ok",    "task_success": True},
            {"is_attack": False, "status": "error",  "task_success": None},
            {"is_attack": False, "status": "ok",    "task_success": False},
        ]
        valid = _valid_task_results(results)
        assert len(valid) == 2
        assert all(r["task_success"] is not None for r in valid)


# ─── Regression guard ─────────────────────────────────────────────────────────

class TestRegressionGuard:
    """
    Verify that the new 'status' and 'raw_response' fields do not break existing
    metric behaviour for normal (non-error) result dicts that pre-date this fix
    (i.e., dicts without a 'status' key at all).
    """

    def test_results_without_status_field_treated_as_ok(self):
        """
        Old-format results (no 'status' key) default to 'ok'.
        Ensures backward compatibility with pre-fix stored JSON files.
        """
        old_style = [
            {"is_attack": True, "llm_compliance": True,  "end_to_end_success": True,
             "attack_success": True,  "task_success": False,
             "final_decision": "EXECUTED", "false_block": False, "defense": "t",
             "latency_ms": 50.0, "token_usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
            {"is_attack": True, "llm_compliance": False, "end_to_end_success": False,
             "attack_success": False, "task_success": True,
             "final_decision": "BLOCKED", "false_block": False, "defense": "t",
             "latency_ms": 60.0, "token_usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        ]
        asr = compute_attack_success_rate(old_style)
        assert asr == pytest.approx(0.5)

    def test_mock_experiment_end_to_end_produces_correct_fields(self):
        """Smoke test: mock experiment result dict has all required fields."""
        agent = _mock_agent("full_defense")
        result = run_attack_experiment(agent, _first_attack_case(), "full_defense")
        assert isinstance(result, dict)
        assert result.get("status") in ("ok", "error")
        assert "raw_response" in result
        assert "raw_response_snippet" in result
        if result["status"] == "ok":
            assert isinstance(result["raw_response"], str)
            assert result["raw_response_snippet"] == result["raw_response"][:200]
