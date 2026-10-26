import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.llm import GeminiProvider, LLMProviderError, LLMTimeoutError

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
load_dotenv()


class ChatProvider(Protocol):
    async def generate(
        self, messages: list[dict[str, str]], system_prompt: str
    ) -> str: ...


@dataclass
class Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must not be blank.")
        return value.strip()


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    messages: list[dict[str, str]]


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: list[dict[str, str]]


def create_app(
    settings: Settings | None = None,
    provider_factory: Callable[[str, str, float], ChatProvider] = GeminiProvider,
) -> FastAPI:
    configured = settings or Settings.from_env()
    app = FastAPI(title="Context Chatbot", version="1.0.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    provider = (
        provider_factory(
            configured.api_key, configured.model, configured.llm_timeout_seconds
        )
        if configured.api_key
        else None
    )
    conversations: dict[str, Conversation] = {}
    conversations_lock = asyncio.Lock()

    async def get_conversation(conversation_id: str | None) -> tuple[str, Conversation]:
        async with conversations_lock:
            if conversation_id is None:
                conversation_id = str(uuid4())
                conversations[conversation_id] = Conversation()
            conversation = conversations.get(conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found.")
            return conversation_id, conversation

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
    async def read_conversation(conversation_id: str) -> ConversationResponse:
        _, conversation = await get_conversation(conversation_id)
        async with conversation.lock:
            return ConversationResponse(
                conversation_id=conversation_id,
                messages=[message.copy() for message in conversation.messages],
            )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if provider is None:
            raise HTTPException(
                status_code=503,
                detail="Chat is not configured. Set GOOGLE_API_KEY in your environment.",
            )

        conversation_id = request.conversation_id
        async with conversations_lock:
            if conversation_id is None:
                conversation_id = str(uuid4())
                conversation = Conversation()
                is_new_conversation = True
            else:
                is_new_conversation = False
                conversation = conversations.get(conversation_id)
                if conversation is None:
                    raise HTTPException(
                        status_code=404, detail="Conversation not found."
                    )

        async with conversation.lock:
            pending = conversation.messages + [
                {"role": "user", "content": request.message.strip()}
            ]
            context = pending[-configured.context_max_messages :]
            if context[0]["role"] == "assistant":
                context = context[1:]
            try:
                reply = await asyncio.wait_for(
                    provider.generate(context, configured.system_prompt),
                    timeout=configured.llm_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise HTTPException(
                    status_code=504,
                    detail="The model request timed out. Please try again.",
                ) from exc
            except LLMTimeoutError as exc:
                raise HTTPException(status_code=504, detail=str(exc)) from exc
            except LLMProviderError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("Unexpected chatbot provider error")
                raise HTTPException(
                    status_code=502,
                    detail="The model service is temporarily unavailable. Please try again.",
                ) from exc

            conversation.messages.extend(
                [
                    {"role": "user", "content": request.message.strip()},
                    {"role": "assistant", "content": reply},
                ]
            )
            if is_new_conversation:
                async with conversations_lock:
                    conversations[conversation_id] = conversation
            return ChatResponse(
                conversation_id=conversation_id,
                reply=reply,
                messages=[message.copy() for message in conversation.messages],
            )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
