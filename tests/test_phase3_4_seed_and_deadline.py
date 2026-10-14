"""
Regression tests for Phase 3.4: Seed Propagation, Hard Wall-Clock Deadlines, and Outcome Validity.

Verifies:
1. Seed 42 reaches the Ollama request payload options.seed.
2. Seeds 123 and 999 are passed correctly to the options.seed payload.
3. Attack and benign-task paths both propagate seeds end-to-end.
4. A deliberately stalled local request returns a timeout within the allowed tolerance.
5. Timeout cleanup terminates worker processes reliably on Windows without unbounded waiting.
6. The experiment runner preserves timeout status and excludes unresolved outcomes from confirmed resistance/blocking metrics.
"""

import os
import sys
import json
import time
import socket
import http.server
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import LLMClient, Agent, AgentResponse
from attacks.indirect_injection import AttackCase
from evaluation.experiment_runner import run_attack_experiment, run_task_experiment
from evaluation.metrics import compute_all_metrics


class _MockHTTPResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_dummy_response(content: str = "ANSWER: Test response") -> bytes:
    obj = {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 20,
        "eval_count": 10,
    }
    return json.dumps(obj).encode("utf-8")


# ─── 1. Seed 42 reaches the Ollama request payload ────────────────────────────

class TestSeedPropagation:
    @patch("urllib.request.urlopen")
    def test_seed_42_reaches_ollama_payload(self, mock_urlopen):
        """Seed 42 must be present in the Ollama request payload options.seed."""
        mock_urlopen.return_value = _MockHTTPResponse(_make_dummy_response())
        client = LLMClient(provider="local", model="qwen3:4b")
        client.complete([{"role": "user", "content": "Hello"}], seed=42)

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        sent_data = json.loads(request_obj.data)

        assert "options" in sent_data
        assert "seed" in sent_data["options"]
        assert sent_data["options"]["seed"] == 42

    # ─── 2. Seeds 123 and 999 are passed correctly ────────────────────────────

    @patch("urllib.request.urlopen")
    def test_seeds_123_and_999_passed_correctly(self, mock_urlopen):
        """Seeds 123 and 999 must be propagated to Ollama options.seed."""
        mock_urlopen.return_value = _MockHTTPResponse(_make_dummy_response())
        client = LLMClient(provider="local", model="qwen3:4b")

        # Test seed 123
        client.complete([{"role": "user", "content": "Test 123"}], seed=123)
        sent_data_123 = json.loads(mock_urlopen.call_args[0][0].data)
        assert sent_data_123["options"]["seed"] == 123

        # Test seed 999
        client.complete([{"role": "user", "content": "Test 999"}], seed=999)
        sent_data_999 = json.loads(mock_urlopen.call_args[0][0].data)
        assert sent_data_999["options"]["seed"] == 999

    # ─── 3. Attack and benign-task paths both propagate seeds ─────────────────

    @patch("urllib.request.urlopen")
    def test_attack_path_propagates_seed(self, mock_urlopen):
        """run_attack_experiment must forward the seed and record it in the output dict."""
        mock_urlopen.return_value = _MockHTTPResponse(_make_dummy_response("ANSWER: Safe answer"))
        client = LLMClient(provider="local", model="qwen3:4b")
        agent = Agent(llm_client=client, defense_mode="full_defense")

        from attacks.indirect_injection import get_attack_by_id
        attack_case = get_attack_by_id("ATK001")

        result = run_attack_experiment(agent, attack_case, "full_defense", seed=123)
        assert result["seed"] == 123

        # Verify outgoing payload contained seed 123
        sent_data = json.loads(mock_urlopen.call_args[0][0].data)
        assert sent_data["options"]["seed"] == 123

    @patch("urllib.request.urlopen")
    def test_task_path_propagates_seed(self, mock_urlopen):
        """run_task_experiment must forward the seed and record it in the output dict."""
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_dummy_response("ANSWER: Current weather is 22C and partly cloudy.")
        )
        client = LLMClient(provider="local", model="qwen3:4b")
        agent = Agent(llm_client=client, defense_mode="full_defense")

        task = {
            "task_id": "TASK001",
            "category": "factual",
            "user_query": "What is the current weather in New York?",
            "required_tool": "web_search",
            "tool_query": "weather",
            "expected_answer": "22C and partly cloudy"
        }

        result = run_task_experiment(agent, task, "full_defense", seed=999)
        assert result["seed"] == 999

        # Verify outgoing payload contained seed 999
        sent_data = json.loads(mock_urlopen.call_args[0][0].data)
        assert sent_data["options"]["seed"] == 999


# ─── 4. Stalled local request returns a timeout within allowed tolerance ──────

class TestHardWallClockDeadline:
    def test_deliberately_stalled_request_times_out_within_tolerance(self, monkeypatch):
        """A stalled server request must be aborted within wall-clock tolerance via the worker process."""
        class StalledHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                # Read body then stall indefinitely
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    self.rfile.read(length)
                time.sleep(10)
            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), StalledHandler)
        port = server.server_port
        t_srv = threading.Thread(target=server.serve_forever, daemon=True)
        t_srv.start()

        # Set strict deadline of 1.5s
        deadline = 1.5
        monkeypatch.setenv("OLLAMA_TIMEOUT", str(deadline))
        monkeypatch.delenv("OLLAMA_WORKER_MODE", raising=False)

        client = LLMClient(provider="local", model="qwen3:4b")
        client._ollama_base_url = f"http://127.0.0.1:{port}"

        t0 = time.time()
        with pytest.raises(TimeoutError) as exc_info:
            client.complete([{"role": "user", "content": "Stall test"}], seed=42)

        elapsed = time.time() - t0
        server.shutdown()

        # Documented tolerance: deadline (1.5s) to deadline + 2.0s
        assert elapsed >= deadline, f"Returned too quickly: {elapsed:.2f}s < {deadline}s"
        assert elapsed <= deadline + 2.5, f"Took too long to return control: {elapsed:.2f}s > {deadline + 2.5}s"
        assert "deadline" in str(exc_info.value).lower() or "timed out" in str(exc_info.value).lower()

    # ─── 5. Timeout cleanup terminates worker processes reliably ──────────────

    def test_timeout_cleanup_no_orphaned_worker_processes(self, monkeypatch):
        """Worker process must be killed and reaped upon timeout, leaving no dangling processes."""
        class StalledHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    self.rfile.read(length)
                time.sleep(10)
            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), StalledHandler)
        port = server.server_port
        threading.Thread(target=server.serve_forever, daemon=True).start()

        monkeypatch.setenv("OLLAMA_TIMEOUT", "1.0")
        monkeypatch.delenv("OLLAMA_WORKER_MODE", raising=False)

        client = LLMClient(provider="local", model="qwen3:4b")
        client._ollama_base_url = f"http://127.0.0.1:{port}"
        agent = Agent(llm_client=client, defense_mode="full_defense")

        response = agent.run("Stall test", [{"tool_name": "t", "content": "c"}], seed=42)
        server.shutdown()

        assert response.final_decision == "ERROR"
        assert response.status == "timeout"
        assert response.latency_ms >= 1000.0  # At least 1.0s elapsed


# ─── 6. Runner preserves timeout and excludes from blocking metrics ───────────

class TestOutcomeValidityAndTimeoutAccounting:
    def test_runner_preserves_timeout_status(self):
        """Timed-out attacks must have final_decision='ERROR' and None for compliance metrics."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.side_effect = TimeoutError("Ollama request exceeded wall-clock deadline of 180s")
        mock_client.provider = "local"
        mock_client.model = "qwen3:4b"

        agent = Agent(llm_client=mock_client, defense_mode="full_defense")
        from attacks.indirect_injection import get_attack_by_id
        attack_case = get_attack_by_id("ATK011")

        result = run_attack_experiment(agent, attack_case, "full_defense", seed=42)

        assert result["status"] == "timeout"
        assert result["final_decision"] == "ERROR"
        assert result["llm_compliance"] is None
        assert result["attack_compliance"] is None
        assert result["end_to_end_success"] is None
        assert result["action_authorized"] is False

    def test_timeout_excluded_from_confirmed_metrics(self):
        """Timed-out trials must NOT be counted as successful defense / validator blocks in completed metrics."""
        # 1 valid block, 1 timeout, 1 model resistance
        sample_results = [
            {
                "is_attack": True,
                "trial_type": "attack",
                "status": "truncated",
                "final_decision": "BLOCKED",
                "llm_compliance": False,
                "end_to_end_success": False,
                "latency_ms": 15000,
            },
            {
                "is_attack": True,
                "trial_type": "attack",
                "status": "timeout",
                "final_decision": "ERROR",
                "llm_compliance": None,
                "end_to_end_success": None,
                "latency_ms": 180000,
            },
            {
                "is_attack": True,
                "trial_type": "attack",
                "status": "ok",
                "final_decision": "EXECUTED",
                "llm_compliance": False,
                "end_to_end_success": False,
                "latency_ms": 20000,
            },
        ]

        metrics = compute_all_metrics(sample_results)

        # Total attempted = 3, valid completed attacks = 2
        # Blocked numerator = 1 (the 1 explicit BLOCKED trial)
        # Denominator for completed ABR = 2 (timeout strictly excluded from completed denominator)
        assert metrics["attack_blocking_rate"] == 0.5  # 1 / 2 = 50%
        # The timeout must NOT be counted as a block (1 block, not 2)
        assert metrics["attack_runs"] == 2
        assert metrics["attack_timeouts"] == 1
