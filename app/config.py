from dataclasses import dataclass
import os


SYSTEM_PROMPT = (
    "You are a thoughtful, friendly, and concise assistant. Use the conversation "
    "history to understand follow-up questions, maintain context, and be honest "
    "when you do not know something."
)


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    context_max_messages: int
    llm_timeout_seconds: float
    system_prompt: str = SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> "Settings":
        max_messages = int(os.getenv("CHAT_CONTEXT_MAX_MESSAGES", "20"))
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        if not 2 <= max_messages <= 200:
            raise ValueError("CHAT_CONTEXT_MAX_MESSAGES must be between 2 and 200.")
        if not 0 < timeout <= 120:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than 0 and at most 120.")
        return cls(
            api_key=os.getenv("GOOGLE_API_KEY") or None,
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            context_max_messages=max_messages,
            llm_timeout_seconds=timeout,
        )
