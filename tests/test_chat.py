import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import GeminiProvider, LLMProviderError, LLMTimeoutError
from app.main import create_app
from google.genai import errors
from google.genai._gaos.lib.compat_errors import (
    APITimeoutError,
    APIStatusError,
)


class FakeProvider:
    def __init__(self, result="A thoughtful answer.", error=None, wait=None):
        self.result = result
        self.error = error
        self.wait = wait
        self.calls = []

    async def generate(self, messages, system_prompt):
        self.calls.append((messages, system_prompt))
        if self.wait:
            await asyncio.sleep(self.wait)
        if self.error:
            raise self.error
        return self.result


def settings(**overrides):
    values = {
        "api_key": "test-key",
        "model": "test-model",
        "context_max_messages": 4,
        "llm_timeout_seconds": 0.05,
    }
    values.update(overrides)
    return Settings(**values)


def make_client(provider, configured=None):
    app = create_app(
        settings=configured or settings(),
        provider_factory=lambda api_key, model, timeout_seconds: provider,
    )
    return TestClient(app)


def test_conversation_retains_turns_and_trims_only_provider_context():
    provider = FakeProvider()
    with make_client(provider) as client:
        first = client.post("/api/chat", json={"message": "First question"})
        conversation_id = first.json()["conversation_id"]
        second = client.post(
            "/api/chat", json={"message": "Follow up", "conversation_id": conversation_id}
        )

        assert first.status_code == second.status_code == 200
        assert len(second.json()["messages"]) == 4
        sent_context, system_prompt = provider.calls[-1]
        assert sent_context == [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "A thoughtful answer."},
            {"role": "user", "content": "Follow up"},
        ]
        assert "conversation history" in system_prompt

        for text in ("Third", "Fourth"):
            client.post("/api/chat", json={"message": text, "conversation_id": conversation_id})
        last_context, _ = provider.calls[-1]
        assert len(last_context) == 3
        assert [message["content"] for message in last_context] == [
            "Third",
            "A thoughtful answer.",
            "Fourth",
        ]
        history = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
        assert len(history) == 8


def test_conversations_have_distinct_ids_and_unknown_ids_are_404():
    with make_client(FakeProvider()) as client:
        first = client.post("/api/chat", json={"message": "Hello"}).json()
        second = client.post("/api/chat", json={"message": "Hello"}).json()
        assert first["conversation_id"] != second["conversation_id"]
        missing = client.post(
            "/api/chat", json={"message": "Hello", "conversation_id": "not-a-conversation"}
        )
        assert missing.status_code == 404


def test_missing_api_key_is_reported_without_calling_provider():
    provider = FakeProvider()
    with make_client(provider, settings(api_key=None)) as client:
        response = client.post("/api/chat", json={"message": "Hello"})
        assert response.status_code == 503
        assert "GOOGLE_API_KEY" in response.json()["detail"]
        assert provider.calls == []


def test_provider_failure_is_visible_and_does_not_store_partial_turn():
    provider = FakeProvider()
    with make_client(provider) as client:
        initial = client.post("/api/chat", json={"message": "First"})
        conversation_id = initial.json()["conversation_id"]
        provider.error = LLMProviderError("Gemini is unavailable.")
        response = client.post(
            "/api/chat", json={"message": "Second", "conversation_id": conversation_id}
        )

        assert response.status_code == 502
        assert response.json()["detail"] == "Gemini is unavailable."
        history = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
        assert [message["content"] for message in history] == [
            "First",
            "A thoughtful answer.",
        ]


def test_timeout_is_reported_and_does_not_store_partial_turn():
    provider = FakeProvider()
    with make_client(provider) as client:
        initial = client.post("/api/chat", json={"message": "First"})
        conversation_id = initial.json()["conversation_id"]
        provider.wait = 0.2
        response = client.post(
            "/api/chat", json={"message": "Second", "conversation_id": conversation_id}
        )
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"]
        history = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
        assert len(history) == 2


def test_input_validation_rejects_empty_or_oversized_message():
    with make_client(FakeProvider()) as client:
        assert client.post("/api/chat", json={"message": ""}).status_code == 422
        assert client.post("/api/chat", json={"message": "   "}).status_code == 422
        assert client.post("/api/chat", json={"message": "x" * 4001}).status_code == 422


def test_invalid_gemini_key_error_explains_how_to_fix_configuration():
    sdk_client = MagicMock()
    sdk_client.aio.interactions.create = AsyncMock(
        side_effect=errors.ClientError(
            400,
            {
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "details": [{"reason": "API_KEY_INVALID"}],
                }
            },
        )
    )
    with patch("app.llm.genai.Client", return_value=sdk_client):
        provider = GeminiProvider("never-printed-test-key", "test-model")
        try:
            asyncio.run(provider.generate([], "system prompt"))
        except LLMProviderError as exc:
            assert "Google rejected the Gemini API key" in str(exc)
            assert ".env file" in str(exc)
            assert "never-printed-test-key" not in str(exc)
        else:
            raise AssertionError("Expected invalid API key to be reported")


def test_gemini_network_timeout_is_reported_as_timeout():
    sdk_client = MagicMock()
    sdk_client.aio.interactions.create = AsyncMock(
        side_effect=APITimeoutError(httpx.Request("POST", "https://example.test"))
    )
    with patch("app.llm.genai.Client", return_value=sdk_client):
        provider = GeminiProvider("never-printed-test-key", "test-model")
        try:
            asyncio.run(provider.generate([], "system prompt"))
        except LLMTimeoutError as exc:
            assert "internet connection" in str(exc)
            assert "never-printed-test-key" not in str(exc)
        else:
            raise AssertionError("Expected provider timeout to be reported")


def test_unavailable_model_error_is_actionable():
    sdk_client = MagicMock()
    body = {
        "error": {
            "code": 404,
            "status": "NOT_FOUND",
            "message": "Configured model is unavailable",
        }
    }
    sdk_client.aio.interactions.create = AsyncMock(
        side_effect=APIStatusError(
            "Model unavailable",
            response=httpx.Response(
                404, request=httpx.Request("POST", "https://example.test")
            ),
            body=body,
        )
    )
    with patch("app.llm.genai.Client", return_value=sdk_client):
        provider = GeminiProvider("never-printed-test-key", "test-model")
        try:
            asyncio.run(provider.generate([], "system prompt"))
        except LLMProviderError as exc:
            assert "GEMINI_MODEL" in str(exc)
            assert "never-printed-test-key" not in str(exc)
        else:
            raise AssertionError("Expected unavailable model to be reported")


def test_gemini_uses_interactions_api_with_roles_and_private_storage_disabled():
    sdk_client = MagicMock()
    sdk_client.aio.interactions.create = AsyncMock(
        return_value=MagicMock(output_text="Hello there.")
    )
    with patch("app.llm.genai.Client", return_value=sdk_client):
        provider = GeminiProvider("never-printed-test-key", "gemini-3.8-flash")
        answer = asyncio.run(
            provider.generate(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Follow up"},
                ],
                "Be helpful.",
            )
        )

    assert answer == "Hello there."
    sdk_client.aio.interactions.create.assert_awaited_once_with(
        model="gemini-3.8-flash",
        input="User: Hi\nAssistant: Hello\nUser: Follow up",
        system_instruction="Be helpful.",
        store=False,
        timeout=30,
    )


def test_settings_default_to_available_gemini_model(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert Settings.from_env().model == "gemini-3.8-flash"


def test_frontend_and_health_endpoint_are_served():
    with make_client(FakeProvider()) as client:
        assert "Good conversations" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/health").json() == {"status": "ok"}
