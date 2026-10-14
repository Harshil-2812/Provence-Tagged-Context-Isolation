"""
Tests for the local (Ollama) provider integration in LLMClient.

No real Ollama calls are made. All HTTP calls are mocked via unittest.mock.
Tests verify:
  1. Provider initialization and model name
  2. Correct message format sent to Ollama /api/chat
  3. Successful response and token usage extraction
  4. Qwen3 thinking-tag stripping (</think> content removed)
  5. Ollama unavailable / connection error handling
  6. Response format compatibility with the experiment runner
  7. Mock provider remains unaffected
"""

import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import LLMClient, Agent, AgentResponse

# ─── Sample message fixtures ─────────────────────────────────────────────────

SAMPLE_MESSAGES = [
    {"role": "system",  "content": "You are a helpful assistant."},
    {"role": "user",    "content": "What is the capital of France?"},
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_ollama_response(content: str, prompt_eval_count: int = 40,
                           eval_count: int = 12) -> bytes:
    """Return bytes that look like an Ollama /api/chat response."""
    obj = {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "total_duration": 1_000_000_000,
        "prompt_eval_duration": 100_000_000,
        "eval_duration": 900_000_000,
    }
    return json.dumps(obj).encode()


class _MockHTTPResponse:
    """Minimal mock for urllib.request.urlopen return value."""
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ─── 1. Provider initialization ──────────────────────────────────────────────

class TestLocalProviderInit:
    def test_local_provider_accepted(self):
        """LLMClient('local') must not raise ValueError."""
        client = LLMClient(provider="local")
        assert client.provider == "local"

    def test_default_model_is_qwen3_4b(self):
        """Default model must be qwen3:4b when LOCAL_MODEL env var is unset."""
        os.environ.pop("LOCAL_MODEL", None)
        client = LLMClient(provider="local")
        assert client.model == "qwen3:4b"

    def test_model_from_env(self):
        """LOCAL_MODEL env var must override the default model name."""
        os.environ["LOCAL_MODEL"] = "llama3:8b"
        client = LLMClient(provider="local")
        os.environ.pop("LOCAL_MODEL", None)
        assert client.model == "llama3:8b"

    def test_base_url_defaults_to_localhost(self):
        """Ollama base URL must default to http://localhost:11434."""
        os.environ.pop("OLLAMA_BASE_URL", None)
        client = LLMClient(provider="local")
        assert client._ollama_base_url == "http://localhost:11434"

    def test_base_url_from_env(self):
        """OLLAMA_BASE_URL env var must override the default URL."""
        os.environ["OLLAMA_BASE_URL"] = "http://remote:11434"
        client = LLMClient(provider="local")
        os.environ.pop("OLLAMA_BASE_URL", None)
        assert client._ollama_base_url == "http://remote:11434"

    def test_no_api_key_required(self):
        """Local provider must not require an api_key parameter."""
        client = LLMClient(provider="local", api_key=None)
        assert client.provider == "local"

    def test_unknown_provider_still_raises(self):
        """Unrecognised provider name must still raise ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMClient(provider="gemini_old_broken")

    def test_local_in_valid_providers_message(self):
        """ValueError message for unknown providers must list 'local'."""
        with pytest.raises(ValueError) as exc_info:
            LLMClient(provider="bad_provider")
        assert "local" in str(exc_info.value).lower()


# ─── 2. Successful response ───────────────────────────────────────────────────

class TestLocalProviderSuccess:
    def _client(self, model="qwen3:4b"):
        os.environ.pop("LOCAL_MODEL", None)
        return LLMClient(provider="local", model=model)

    @patch("urllib.request.urlopen")
    def test_returns_response_text(self, mock_urlopen):
        """complete() must return the model text as the first element."""
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response("ANSWER: Paris is the capital of France.")
        )
        client = self._client()
        text, _ = client.complete(SAMPLE_MESSAGES)
        assert text == "ANSWER: Paris is the capital of France."

    @patch("urllib.request.urlopen")
    def test_returns_token_usage_dict(self, mock_urlopen):
        """complete() must return a usage dict with three standard keys."""
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response("ANSWER: Paris.", prompt_eval_count=40, eval_count=12)
        )
        client = self._client()
        _, usage = client.complete(SAMPLE_MESSAGES)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    @patch("urllib.request.urlopen")
    def test_token_counts_correct(self, mock_urlopen):
        """Token counts must match Ollama's prompt_eval_count / eval_count fields."""
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response("ANSWER: Paris.", prompt_eval_count=55, eval_count=15)
        )
        client = self._client()
        _, usage = client.complete(SAMPLE_MESSAGES)
        assert usage["prompt_tokens"] == 55
        assert usage["completion_tokens"] == 15
        assert usage["total_tokens"] == 70

    @patch("urllib.request.urlopen")
    def test_model_name_sent_to_ollama(self, mock_urlopen):
        """The model name must be forwarded to Ollama in the request payload."""
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response("ANSWER: Paris.")
        )
        client = self._client(model="qwen3:4b")
        client.complete(SAMPLE_MESSAGES)
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        sent_data = json.loads(request_obj.data)
        assert sent_data["model"] == "qwen3:4b"

    @patch("urllib.request.urlopen")
    def test_messages_forwarded_to_ollama(self, mock_urlopen):
        """All messages must be forwarded to Ollama with correct roles."""
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response("ANSWER: Paris.")
        )
        client = self._client()
        client.complete(SAMPLE_MESSAGES)
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        sent_data = json.loads(request_obj.data)
        roles = [m["role"] for m in sent_data["messages"]]
        assert "system" in roles
        assert "user" in roles


# ─── 3. Qwen3 thinking-tag stripping ─────────────────────────────────────────

class TestQwen3ThinkingTagStripping:
    """</think> content must be stripped; only the final answer is returned."""

    @patch("urllib.request.urlopen")
    def test_think_tag_stripped(self, mock_urlopen):
        """Content before and including </think> must be removed."""
        raw_content = (
            "Okay, the user asked about France...\n\nSome more thinking.\n</think>\n"
            "ANSWER: Paris is the capital of France."
        )
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response(raw_content)
        )
        client = LLMClient(provider="local", model="qwen3:4b")
        text, _ = client.complete(SAMPLE_MESSAGES)
        assert "<think>" not in text
        assert "</think>" not in text
        assert text == "ANSWER: Paris is the capital of France."

    @patch("urllib.request.urlopen")
    def test_no_think_tag_returned_unchanged(self, mock_urlopen):
        """If no </think> is present, content must be returned as-is."""
        raw_content = "ANSWER: Paris is the capital of France."
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response(raw_content)
        )
        client = LLMClient(provider="local", model="qwen3:4b")
        text, _ = client.complete(SAMPLE_MESSAGES)
        assert text == raw_content

    @patch("urllib.request.urlopen")
    def test_answer_prefix_preserved_after_strip(self, mock_urlopen):
        """ANSWER: prefix must survive the stripping process."""
        raw_content = "Some reasoning\n</think>\nANSWER: 42"
        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response(raw_content)
        )
        client = LLMClient(provider="local", model="qwen3:4b")
        text, _ = client.complete(SAMPLE_MESSAGES)
        assert text.startswith("ANSWER:")


# ─── 4. Error handling ────────────────────────────────────────────────────────

class TestLocalProviderErrorHandling:
    """Ollama unavailable / connection errors must be handled gracefully."""

    @patch("urllib.request.urlopen")
    def test_connection_error_raises_runtime_error(self, mock_urlopen):
        """URLError (Ollama down) must raise RuntimeError with descriptive message."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        client = LLMClient(provider="local")
        with pytest.raises(RuntimeError, match="Ollama service unavailable"):
            client.complete(SAMPLE_MESSAGES)

    @patch("urllib.request.urlopen")
    def test_connection_error_propagates_to_agent(self, mock_urlopen):
        """Agent.run() must catch the error and return final_decision='ERROR'."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        llm = LLMClient(provider="local")
        agent = Agent(llm_client=llm, defense_mode="no_defense")
        response = agent.run(
            user_task="What is the capital of France?",
            tool_outputs=[{"tool_name": "web_search", "content": "Paris is the capital."}],
        )
        assert response.final_decision == "ERROR"
        assert response.error is not None

    @patch("urllib.request.urlopen")
    def test_generic_error_propagates(self, mock_urlopen):
        """Any unexpected exception must propagate (not silently swallowed)."""
        mock_urlopen.side_effect = Exception("unexpected failure")
        client = LLMClient(provider="local")
        with pytest.raises(Exception, match="unexpected failure"):
            client.complete(SAMPLE_MESSAGES)

    @patch("urllib.request.urlopen")
    def test_error_not_counted_as_successful_defense(self, mock_urlopen):
        """A connection error must not be misclassified as a successful defense."""
        import urllib.error
        from evaluation.experiment_runner import run_attack_experiment
        from attacks.indirect_injection import load_attack_cases

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        llm = LLMClient(provider="local")
        agent = Agent(llm_client=llm, defense_mode="no_defense")
        attack_case = load_attack_cases()[0]

        result = run_attack_experiment(agent, attack_case, "no_defense")
        assert result["status"] == "error"
        assert result["llm_compliance"] is None
        assert result["end_to_end_success"] is None

    @patch("urllib.request.urlopen")
    def test_error_not_counted_as_successful_task(self, mock_urlopen):
        """A connection error must not inflate TSR."""
        import urllib.error
        from evaluation.experiment_runner import run_task_experiment, load_task_cases

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        llm = LLMClient(provider="local")
        agent = Agent(llm_client=llm, defense_mode="no_defense")
        task = load_task_cases()[0]

        result = run_task_experiment(agent, task, "no_defense")
        assert result["status"] == "error"
        assert result["task_success"] is None


# ─── 5. Response format compatibility with experiment runner ─────────────────

class TestResponseFormatCompatibility:
    """Verify local provider outputs are compatible with the experiment runner."""

    @patch("urllib.request.urlopen")
    def test_attack_result_has_all_required_fields(self, mock_urlopen):
        """Attack result dict from local provider must have all required keys."""
        from evaluation.experiment_runner import run_attack_experiment
        from attacks.indirect_injection import load_attack_cases

        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response("ANSWER: I will follow the instructions found in the tool output.")
        )
        llm = LLMClient(provider="local")
        agent = Agent(llm_client=llm, defense_mode="no_defense")
        result = run_attack_experiment(agent, load_attack_cases()[0], "no_defense")

        assert "status" in result
        assert "raw_response" in result
        assert "latency_ms" in result
        assert "token_usage" in result
        assert isinstance(result["raw_response"], str)

    @patch("urllib.request.urlopen")
    def test_task_result_has_all_required_fields(self, mock_urlopen):
        """Task result dict from local provider must have all required keys."""
        from evaluation.experiment_runner import run_task_experiment, load_task_cases

        mock_urlopen.return_value = _MockHTTPResponse(
            _make_ollama_response(
                "ANSWER: Paris is the capital of France.",
                prompt_eval_count=40, eval_count=12,
            )
        )
        llm = LLMClient(provider="local")
        agent = Agent(llm_client=llm, defense_mode="no_defense")
        result = run_task_experiment(agent, load_task_cases()[0], "no_defense")

        assert "status" in result
        assert "raw_response" in result
        assert "task_success" in result
        assert isinstance(result["latency_ms"], float)


# ─── 6. Other providers unaffected ───────────────────────────────────────────

class TestExistingProvidersUnaffected:
    def test_mock_provider_still_works(self):
        """Mock provider must still initialise and produce a string response."""
        llm = LLMClient(provider="mock")
        text, usage = llm.complete(SAMPLE_MESSAGES)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "prompt_tokens" in usage

    def test_openai_provider_still_accepted(self):
        """'openai' must still be a valid provider string."""
        with patch("openai.OpenAI"):
            client = LLMClient(provider="openai", api_key="sk-fake")
        assert client.provider == "openai"

    def test_anthropic_provider_still_accepted(self):
        """'anthropic' must still be a valid provider string."""
        with patch("anthropic.Anthropic"):
            client = LLMClient(provider="anthropic", api_key="sk-ant-fake")
        assert client.provider == "anthropic"

    def test_mock_complete_not_routed_to_ollama(self):
        """Mock provider must never call urllib.request.urlopen."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            llm = LLMClient(provider="mock")
            llm.complete(SAMPLE_MESSAGES)
        mock_urlopen.assert_not_called()

    def test_gemini_provider_now_invalid(self):
        """'gemini' must now be an invalid provider (removed)."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMClient(provider="gemini")
