# Context Chatbot

A small, local-first chat app built with FastAPI and Google's official Gemini SDK. Each conversation gets its own ID and retains its complete user/assistant history in memory for display and follow-up messages. Before calling Gemini, the app sends at most the configured number of most recent messages; if that boundary would start on an assistant reply, it skips that reply so the model context begins with a user message. Older messages remain in the conversation history but are not included in the model context.

## Run locally

1. Use Python 3.10 or newer and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy the example environment file and add your own Google AI Studio API key to `.env`:

   ```powershell
   Copy-Item .env.example .env
   ```

   Set `GOOGLE_API_KEY` in `.env` to a key you obtained from [Google AI Studio](https://aistudio.google.com/app/apikey). The example contains only a placeholder; never commit `.env`.

3. Start the server:

   ```powershell
   uvicorn app.main:app --reload
   ```

4. Open <http://127.0.0.1:8000>. The chat UI keeps the current conversation ID in browser local storage and reloads its history while the server process is still running.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | Gemini API key. Chat returns a clear `503` until configured. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model name, called through Google's Interactions API. |
| `CHAT_CONTEXT_MAX_MESSAGES` | `20` | Number of latest messages sent to the model per request (2–200). |
| `LLM_TIMEOUT_SECONDS` | `30` | Provider request timeout (greater than 0, up to 120 seconds). |

Conversation history is held in process memory, so it is isolated by conversation ID but is lost when the server restarts. The model receives a concise assistant personality/system instruction on every call. API failures return readable `502`/`503` errors and timeouts return `504`; failed requests are not added as incomplete conversation turns.

## API

- `POST /api/chat` — body: `{"message":"Hello","conversation_id":null}`. Omit or set the ID to `null` to create a conversation. Returns its ID, the model reply, and full conversation history.
- `GET /api/conversations/{conversation_id}` — retrieve full in-memory conversation history.
- `GET /api/health` — process health check.

## Tests

Run the focused test suite without a Gemini key:

```powershell
py -m pytest -q
```
