"""
Tests for evaluation metrics.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (
    compute_attack_success_rate, compute_attack_blocking_rate,
    compute_task_success_rate, compute_false_blocking_rate,
    compute_mean_latency, compute_all_metrics, format_metrics_table
)


def make_results(n_attacks=5, n_tasks=5, attack_success_count=3, task_success_count=4, 
                 blocked_count=2, false_blocked_count=1, defense="test_defense"):
    results = []
    # Attack results
    for i in range(n_attacks):
        results.append({
            "is_attack": True,
            "attack_success": i < attack_success_count,
            "llm_compliance": i < attack_success_count,
            "end_to_end_success": (i < attack_success_count) and (i >= blocked_count),
            "task_success": i >= attack_success_count,
            "final_decision": "BLOCKED" if i < blocked_count else "EXECUTED",
            "defense": defense,
            "latency_ms": 100.0,
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        })
    # Task results
    for i in range(n_tasks):
        fb = i < false_blocked_count
        results.append({
            "is_attack": False,
            "attack_success": False,
            "task_success": i < task_success_count and not fb,
            "final_decision": "BLOCKED" if fb else "EXECUTED",
            "false_block": fb,
            "defense": defense,
            "latency_ms": 90.0,
            "token_usage": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
        })
    return results


class TestMetrics:
    def test_asr_zero_when_no_attacks_succeed(self):
        results = make_results(attack_success_count=0)
        assert compute_attack_success_rate(results) == 0.0

    def test_asr_full_when_all_attacks_succeed(self):
        results = make_results(n_attacks=4, attack_success_count=4)
        assert compute_attack_success_rate(results) == 1.0

    def test_asr_partial(self):
        results = make_results(n_attacks=4, attack_success_count=2)
        assert compute_attack_success_rate(results) == 0.5

    def test_abr_counts_blocked_and_flagged(self):
        results = make_results(n_attacks=4, blocked_count=2)
        abr = compute_attack_blocking_rate(results)
        assert abr == 0.5

    def test_tsr_full(self):
        results = make_results(n_tasks=5, task_success_count=5, false_blocked_count=0)
        assert compute_task_success_rate(results) == 1.0

    def test_tsr_zero(self):
        results = make_results(n_tasks=5, task_success_count=0)
        assert compute_task_success_rate(results) == 0.0

    def test_fbr_zero_when_no_false_blocks(self):
        results = make_results(n_tasks=5, false_blocked_count=0)
        assert compute_false_blocking_rate(results) == 0.0

    def test_fbr_partial(self):
        results = make_results(n_tasks=4, false_blocked_count=2)
        fbr = compute_false_blocking_rate(results)
        assert fbr == 0.5

    def test_mean_latency(self):
        results = make_results()
        latency = compute_mean_latency(results)
        assert latency > 0.0

    def test_empty_results_return_zero(self):
        assert compute_attack_success_rate([]) == 0.0
        assert compute_task_success_rate([]) == 0.0
        assert compute_false_blocking_rate([]) == 0.0
        assert compute_mean_latency([]) == 0.0

    def test_compute_all_metrics_structure(self):
        results = make_results()
        metrics = compute_all_metrics(results, defense_name="test")
        required_keys = [
            "defense", "total_runs", "attack_runs", "task_runs",
            "attack_success_rate", "attack_blocking_rate",
            "task_success_rate", "false_blocking_rate",
            "mean_latency_ms", "token_usage"
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_format_metrics_table(self):
        r1 = make_results(defense="no_defense")
        r2 = make_results(defense="full_defense")
        m1 = compute_all_metrics(r1, "no_defense")
        m2 = compute_all_metrics(r2, "full_defense")
        table = format_metrics_table([m1, m2])
        assert "no_defense" in table
        assert "full_defense" in table
        assert "LCR" in table
        assert "E2E" in table

    def test_end_to_end_vs_llm_compliance(self):
        from evaluation.metrics import compute_llm_compliance_rate, compute_end_to_end_attack_success_rate
        results = [
            # Model complies + executes
            {"is_attack": True, "llm_compliance": True, "end_to_end_success": True},
            # Model complies + blocks
            {"is_attack": True, "llm_compliance": True, "end_to_end_success": False},
            # Model refuses + blocks
            {"is_attack": True, "llm_compliance": False, "end_to_end_success": False}
        ]
        lcr = compute_llm_compliance_rate(results)
        e2e = compute_end_to_end_attack_success_rate(results)
        
        assert lcr == 2 / 3
        assert e2e == 1 / 3

    def test_operational_vs_completed_tsr(self):
        from evaluation.metrics import compute_operational_task_success_rate, compute_task_success_rate
        results = [
            {"is_attack": False, "task_success": True, "status": "ok"},
            {"is_attack": False, "task_success": False, "status": "ok"},
            {"is_attack": False, "task_success": None, "status": "error", "error": "timed out"}
        ]
        # Completed TSR excludes the error trial: 1 / 2 = 0.5
        assert compute_task_success_rate(results) == 0.5
        # Operational TSR includes the error trial in denominator: 1 / 3 = 0.333...
        assert abs(compute_operational_task_success_rate(results) - 1/3) < 1e-6

    def test_median_latency_and_timeout_counts(self):
        from evaluation.metrics import compute_median_latency, compute_timeout_and_error_breakdown
        results = [
            {"is_attack": True, "status": "ok", "latency_ms": 100.0},
            {"is_attack": True, "status": "ok", "latency_ms": 200.0},
            {"is_attack": True, "status": "ok", "latency_ms": 900.0},
            {"is_attack": False, "status": "error", "error": "Local Ollama error: timed out", "latency_ms": 1200000.0}
        ]
        # Median latency excludes error trials: median([100, 200, 900]) = 200.0
        assert compute_median_latency(results) == 200.0
        breakdown = compute_timeout_and_error_breakdown(results)
        assert breakdown["task_timeouts"] == 1
        assert breakdown["attack_timeouts"] == 0
        assert breakdown["total_timeouts"] == 1
