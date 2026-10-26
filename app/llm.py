import asyncio
import logging
from collections.abc import Sequence

from google import genai
from google.genai import errors
from google.genai._gaos.lib.compat_errors import (
    APIError as InteractionsAPIError,
    APITimeoutError as InteractionsAPITimeoutError,
)
import httpx

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """A safe-to-display error returned by the model provider."""


class LLMTimeoutError(LLMProviderError):
    """The model provider did not respond before its network timeout."""


class GeminiProvider:
    def __init__(
        self, api_key: str, model: str, timeout_seconds: float = 30
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(
        self, messages: Sequence[dict[str, str]], system_prompt: str
    ) -> str:
        transcript = "\n".join(
            f"{'Assistant' if message['role'] == 'assistant' else 'User'}: "
            f"{message['content']}"
            for message in messages
        )
        try:
            response = await self.client.aio.interactions.create(
                model=self.model,
                input=transcript,
                system_instruction=system_prompt,
                store=False,
                timeout=self.timeout_seconds,
            )
        except (
            asyncio.TimeoutError,
            httpx.TimeoutException,
            InteractionsAPITimeoutError,
        ) as exc:
            logger.warning("Gemini request timed out while waiting for a response.")
            raise LLMTimeoutError(
                "Gemini did not respond before the network timeout. Check your "
                "internet connection and try again."
            ) from exc
        except (errors.APIError, InteractionsAPIError) as exc:
            details = getattr(exc, "details", None) or getattr(exc, "body", None)
            error = details.get("error", {}) if isinstance(details, dict) else {}
            reasons = {
                detail.get("reason")
                for detail in error.get("details", [])
                if isinstance(detail, dict)
            }
            if "API_KEY_INVALID" in reasons:
                logger.warning("Gemini rejected the configured API key as invalid.")
                raise LLMProviderError(
                    "Google rejected the Gemini API key. Create an active key in "
                    "Google AI Studio, replace GOOGLE_API_KEY in your .env file, "
                    "and restart the server."
                ) from exc
            status_code = getattr(exc, "code", None) or getattr(
                exc, "status_code", None
            )
            status = getattr(exc, "status", None)
            if status_code == 404:
                logger.warning("Gemini model was not found (status=%s).", status)
                raise LLMProviderError(
                    "The configured Gemini model is unavailable. Set GEMINI_MODEL "
                    "to a model currently available to your Google AI Studio account."
                ) from exc
            if status_code == 429:
                logger.warning("Gemini request was rate limited (status=%s).", status)
                raise LLMProviderError(
                    "The Gemini API quota or rate limit was reached. Check your "
                    "Google AI Studio usage and try again later."
                ) from exc
            if status_code == 403:
                logger.warning("Gemini rejected API access (status=%s).", status)
                raise LLMProviderError(
                    "Google denied Gemini API access. Check that the API is enabled "
                    "for this key and that its restrictions allow this request."
                ) from exc
            logger.warning(
                "Gemini returned an API error (status=%s, code=%s).",
                status,
                status_code,
            )
            raise LLMProviderError(
                "The Gemini API could not complete the request. Check that your "
                "key and model are enabled, then try again."
            ) from exc
        except httpx.NetworkError as exc:
            logger.warning("Could not connect to the Gemini API (%s).", type(exc).__name__)
            raise LLMProviderError(
                "Could not reach the Gemini API. Check your internet connection "
                "and try again."
            ) from exc
        except Exception as exc:
            logger.error(
                "Gemini request failed unexpectedly (%s).", type(exc).__name__
            )
            raise LLMProviderError(
                "The Gemini service could not complete the request. Please try again."
            ) from exc

        answer = response.output_text
        if not answer or not answer.strip():
            raise LLMProviderError(
                "The Gemini service returned an empty response. Please try again."
            )
        return answer.strip()
