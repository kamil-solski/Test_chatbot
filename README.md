# Test Chatbot

Prototype repository for experimenting with LLM-based conversational systems across multiple providers via LiteLLM proxy, including implementations and tests for chat memory, retrieval-augmented generation (RAG), and reasoning workflows. Vanilla JS frontend + FastAPI backend + LangGraph memory + Redis session storage + nginx.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Vanilla HTML + JS (no bundler) |
| Backend | Python · FastAPI · LangGraph · LiteLLM proxy |
| Memory | Redis (session chat history via LangGraph checkpointer) |
| Proxy | nginx |
| Infra | Docker Compose |


## Current features
LangGraph provides us with infrastructure to implement things like chat-history, Chain-of-thought, etc. but logic for those we must implement ourselves. 

- complex layered chat history (routing) - Attention: what we are doing here is basicaly model memory logic. To be more precise chat history is raw data saved. Memory is what we inject to context and it is basis for reasoning logic.
- litellm to switch smoothly between different llm providers
- token optimization strategy

- simple rag (normaly in production grade it get much more complicated, mostly because of chunking strategies - how to make them universal)

## Structure

```
Test_chatbot/
├── backend/
│   ├── app.py            # FastAPI: POST /api/chat, LangGraph graph
│   ├── helpers.py        # log_usage (CSV), debug logging
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── templates/
│   │   └── index.html    # Chat UI with model selector
│   └── src/
│       └── script.js     # Fetch logic + sessionStorage session ID
├── nginx/
│   └── nginx.conf        # Proxies /api/ → backend:8000
├── litellm_config.yaml   # Model routing config for LiteLLM proxy
├── logs/
│   └── chat_usage.csv    # Per-message token usage log (auto-created)
├── docker-compose.yml
├── .env.example
└── .gitignore
```

## Quick start

### 1. Set up environment
```bash
cp .env.example .env
# Required: add OPENAI_API_KEY
# Optional: add ANTHROPIC_API_KEY to enable Claude models
```

### 2. Docker (full stack)
```bash
docker compose up --build
# → http://localhost:3000
```

### 3. Local dev (no Docker)
```bash
# Start Redis (required for session memory)
docker compose up redis -d

# Backend
cd backend
pip install -r requirements.txt
REDIS_URL=redis://localhost:6379 uvicorn app:app --reload --port 8000

# Frontend — open frontend/templates/index.html with VS Code Live Server
# (script.js auto-detects Live Server and points API calls to localhost:8000)
```

## API

### `POST /api/chat`
```json
{
  "message": "Hello!",
  "session_id": "uuid-string",
  "model": "gpt-4o-mini"
}
```

Response:
```json
{
  "role": "assistant",
  "content": "Hi! How can I help?",
  "model": "gpt-4o-mini",
  "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
}
```

### `GET /health`
Returns `{"status": "ok"}`.

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for GPT models |
| `ANTHROPIC_API_KEY` | — | Required for Claude models |
| `LLM_MODEL` | `gpt-4o-mini` | Default model used if frontend sends none |
| `SYSTEM_PROMPT` | `You are a helpful assistant.` | Instructions prepended to every request |
| `REDIS_URL` | set by Docker | Redis connection string |
| `LOG_FILE` | `logs/chat_usage.csv` | Token usage log path |

## Adding a new model

1. Add an entry to `litellm_config.yaml`:
```yaml
- model_name: my-model           # name used in the UI dropdown
  litellm_params:
    model: provider/model-id     # e.g. anthropic/claude-sonnet-4-5-20250929
    api_key: os.environ/MY_API_KEY
```
2. Add `<option value="my-model">My Model</option>` to `frontend/templates/index.html`
3. `docker compose restart litellm`

## Chat history

1. Can be answered from sliding window + summary?
   → YES → stop (no retrieval)

2. Is it exact / positional / temporal?
   → raw DB (filter-only)

3. Is it bounded (time / topic / doc)?
   → filter → vector

4. Is it open semantic recall?
   → pure vector

5. Else
   → fallback: window + summary

TODO: we migth also improve retrieval so it would fire not only when query reaches beyond window, but also when
- recent messages are too large
- or not precise enough 
Although this improvement is subtle and for now we could skip it.

## Reasoning strategies
Just like for chat-memory we have many strategies for reasoning:

- Chain-of-Thought (the simplest linear reasoning)
- Self-Consistency
- Tree-of-Thoughts
- Graph-of-Thoughts
- Least-to-Most Reasoning
- Reflection / Self-Critique
- Debate / Mutli-Agent Reasoning
- Program-of-Thought / Symbolic Reasoning
- Latent Reasoning
- ReAct (Reasning + Acting: very popular for agentic systems like OpenClaw)

## Testing model switching

### Switching models mid-conversation — important caveat

The model selector controls **which model answers the next message only**. The conversation history stored in Redis is model-agnostic — it is a plain list of user/assistant messages. When you switch models mid-conversation:

- The new model receives the full prior context (all previous turns)
- It will answer in its own style, but it reads everything the previous model said
- Token counts will differ because different models tokenise differently
- If the new model's context window is smaller than the accumulated history, you may get a 502 error


### Testing system prompt changes

1. Edit `SYSTEM_PROMPT` in `.env`
2. Restart the backend: `docker compose restart backend`
3. The new prompt takes effect immediately for all **new** messages — even in existing sessions, because the system prompt is prepended at invocation time, not stored in Redis

### Resetting to default system prompt

Set in `.env`:
```
SYSTEM_PROMPT=You are a helpful assistant.
```
Then restart the backend.

## Logs

Both log files are volume-mounted to `./logs/` on the host and auto-created on first write.

### `logs/chat.log` — always on, one line per message

```json
{"timestamp": "2026-04-20T15:18:44+00:00", "session_id": "90403888-...", "model": "claude-haiku-4-5", "question": "What is prompt engineering?", "prompt_tokens": 370, "completion_tokens": 307, "total_tokens": 677}
```

| Field | Description |
|---|---|
| `timestamp` | UTC ISO-8601 |
| `session_id` | Browser tab UUID — new UUID = new session |
| `model` | Model name as sent by the frontend |
| `question` | The user's message |
| `prompt_tokens` | Input tokens for this turn (includes full history) |
| `completion_tokens` | Output tokens for this turn |
| `total_tokens` | Sum of the above |

### `logs/debug.log` — only written when `DEBUG=true`

Verbose per-request state: startup config, model resolved in graph node, actual model returned by LLM. Useful for verifying LiteLLM routing.

```bash
# Enable in .env
DEBUG=true
# Then restart: docker compose restart backend
```

### Useful commands

```bash
# Stream live chat log. Quick note for early logs before session_tokens (which is cummulative) I used total tokens (which are not, you have to sum all total tokens yourself for given session)
tail -f logs/chat.log

# Stream debug log
tail -f logs/debug.log | python3 -m json.tool

# Query with jq — all claude turns
jq 'select(.model | startswith("claude"))' logs/chat.log

# Token totals per session
jq -r '[.session_id, .model, .total_tokens] | @tsv' logs/chat.log
```

**Note on prompt token growth:** Because LangGraph sends the full conversation history on every turn, `prompt_tokens` grows with each message in a session. This is the baseline to compare against when testing summarisation or RAG-based memory strategies.

**Note on model comparison:** To fairly compare token costs across models, click the clear button before switching models to start a fresh session.


## Example test chat prompt set (when MAX_HISTORY_TOKENS=1000, SUMMARIZE_AFTER=3, RAG_TOP_K=4, WINDOWS_SIZE=6, TOPIC_SIMILARITY_THRESHOLD=0.65)

| #  | Prompt | Assessment |
|----|--------| -----------|
| 1. | Who are you? | Baseline. Also tests that DEFAULT_SYSTEM_PROMPT is injected and model vendor identification |
| 2. | What is prompt engineering? | Anchor. Referenced by #5, #6, #7 |
| 3. | Is PE important? | Test 1-turn memory. |
| 4. | What other skills? | SUMMARIZE_AFTER=3 means after 4 messages (>3 check), so summarization first fires after the response to prompt #2, not #4. By prompt #4 you've already had 2 summarization cycles. |
| 5. | PE becoming less important / context engineering? | Good reasoning test. Minor: this is self-contained enough that even trim with very low MAX_HISTORY_TOKENS would answer it correctly — consider whether you actually want to test that strategy fails here or succeeds. |
| 6. | Provide PE techniques examples. | Correct — by message #11 in the exchange, WINDOW_SIZE=6 cuts off everything before prompt #4. The window no longer contains prompt #2, so full succeeds here while smart/rag must retrieve. |
| 7. | Combine the skills you listed when I asked 'What other skills?' with the PE techniques you described into 3-step roadmap. | Annotation fix needed: "rag extraction" only applies to smart/rag strategies. For summarize, you're testing whether the summary captured prompt #4's content — a different failure mode. For full, it just reads #4 directly. Label it: "tests cross-strategy memory coverage — summary accuracy vs. vector recall vs. full log". |
| 8. | What was the first question? | Correct intent. One correction: "definitely exceeded MAX_HISTORY_TOKENS=1000" depends on response length. With short model responses it may only be ~600-700 tokens by then. Say "likely exceeded" and confirm from the actual prompt_tokens log. For smart, this triggers POSITION intent → the router scans history from LangGraph state, which is full for smart (unlike summarize where RemoveMessage has already deleted early messages). '
| 9. | My favorite language is Python. | fact storage	binary: later recall is correct / wrong |
| 10. | What is my favorite language? | fact retrieval	exact match |
| 11. | What was the first topic we discussed? | positional	exact match against session log | 
| 12. | Summarize what we've covered so far. | compression fidelity	manual: are all topics present? |
| 13. | Earlier I told you my favorite language — what did you say it was? | hallucination probe	binary: fabricated claim vs. none |


## What to experiment with next:
- **Streaming responses** — use `stream=True` in `ChatOpenAI` + SSE on frontend
- **System prompt editor** — add a UI textarea to change the prompt without restarting
- **Tool calling** — add `@tool` decorated functions and `ToolNode` to the LangGraph graph
- **Memory summarisation** — summarise old turns to keep prompt tokens flat as sessions grow
- **RAG** — embed documents and retrieve relevant chunks to inject into the system prompt

## TODO:
1. Implement reasoning (chain-of-thougth, and other)
2. Fix explicit patterns in router with reasoning (intent reasoning) instead regular expression

3. Graph memory

## Useful links:
https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents [Context Engineering]

https://arxiv.org/html/2510.05381v1?utm_source=chatgpt.com [Context Length Alone Hurts LLM Performance Despite Perfect Retrieval]

https://fastpaca.com/blog/llm-memory-systems-explained/?utm_source=chatgpt.com [LLM Memory Systems Explanation]

https://www.c-sharpcorner.com/article/how-llm-memory-works-architecture-techniques-and-developer-patterns/?utm_source=chatgpt.com [How LLM Memory Works: Architecture, Techniques, and Developer Patterns]

https://aiagentmemory.org/articles/how-llm-memory-works/?utm_source=chatgpt.com [How LLM Memory Works: Architectures and Mechanisms]

https://arxiv.org/html/2512.20237v1?utm_source=chatgpt.com [MemR3: Memory Retrieval via Reflective Reasoning for LLM Agents]