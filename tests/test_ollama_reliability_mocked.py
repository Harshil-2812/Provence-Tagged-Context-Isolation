"""
Unit tests for Ollama client reliability with mocked HTTP calls.
Verifies:
- Socket/wall-clock timeouts raise TimeoutError and result in status='timeout'
- Malformed JSON responses raise ValueError and result in status='malformed'
- Service unavailability raises RuntimeError and results in status='error'
- Configurable environment variables LOCAL_MAX_TOKENS and OLLAMA_TIMEOUT
- Token usage extraction including max_tokens_limit
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import urllib.error

from agent.agent import LLMClient, Agent, AgentResponse


class TestOllamaMockedReliability:
    def test_ollama_timeout_maps_to_status_timeout(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_TIMEOUT", "1.0")
        client = LLMClient(provider="local", model="qwen3:4b")
        
        # Mock urllib.request.urlopen raising timeout URLError
        timeout_err = urllib.error.URLError("timed out")
        with patch("urllib.request.urlopen", side_effect=timeout_err):
            with pytest.raises(TimeoutError) as exc_info:
                client.complete([{"role": "user", "content": "Hello"}])
            assert "timed out" in str(exc_info.value).lower()

        agent = Agent(llm_client=client, defense_mode="full_defense")
        with patch("urllib.request.urlopen", side_effect=timeout_err):
            response = agent.run("Hello", [{"tool_name": "t", "content": "c"}])
            assert response.final_decision == "ERROR"
            assert response.status == "timeout"
            assert "timed out" in response.error.lower()

    def test_ollama_malformed_json_maps_to_status_malformed(self):
        client = LLMClient(provider="local", model="qwen3:4b")
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<!DOCTYPE html><html>502 Bad Gateway</html>"
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            agent = Agent(llm_client=client, defense_mode="full_defense")
            response = agent.run("Hello", [{"tool_name": "t", "content": "c"}])
            assert response.final_decision == "ERROR"
            assert response.status == "malformed"
            assert "malformed" in response.error.lower()

    def test_ollama_connection_refusal_maps_to_status_error(self):
        client = LLMClient(provider="local", model="qwen3:4b")
        conn_err = urllib.error.URLError("Connection refused [Errno 111]")

        with patch("urllib.request.urlopen", side_effect=conn_err):
            agent = Agent(llm_client=client, defense_mode="full_defense")
            response = agent.run("Hello", [{"tool_name": "t", "content": "c"}])
            assert response.final_decision == "ERROR"
            assert response.status == "error"
            assert "unavailable" in response.error.lower()

    def test_ollama_custom_env_vars_respected(self, monkeypatch):
        monkeypatch.setenv("LOCAL_MAX_TOKENS", "4096")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "60.0")
        client = LLMClient(provider="local", model="qwen3:4b")

        mock_resp = MagicMock()
        mock_data = {
            "message": {"content": "ANSWER: Paris is the capital."},
            "prompt_eval_count": 150,
            "eval_count": 30,
        }
        mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            text, usage = client.complete([{"role": "user", "content": "What is Paris?"}])
            assert text == "ANSWER: Paris is the capital."
            assert usage["max_tokens_limit"] == 4096
            assert usage["prompt_tokens"] == 150
            assert usage["completion_tokens"] == 30
            
            # Verify timeout passed to urlopen
            _, kwargs = mock_urlopen.call_args
            assert kwargs.get("timeout") == 60.0
