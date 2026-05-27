# Test Chatbot

Prototype repository for experimenting with LLM-based conversational systems across multiple providers via LiteLLM proxy, including implementations and tests for chat memory, retrieval-augmented generation (RAG), and reasoning workflows. Vanilla JS frontend + FastAPI backend + LangGraph memory + Redis session storage + nginx.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Vanilla HTML + JS (no bundler) |
| Backend | Python · FastAPI · LangGraph · LiteLLM proxy |
| Memory | Redis (raw session history via LangGraph checkpointer) + in-memory FAISS (vector retrieval) |
| Proxy | nginx |
| Infra | Docker Compose |


## Current features

LangGraph provides infrastructure (Redis checkpointing, graph execution) but the memory and routing logic is custom-built.

**Terminology distinction:** _Chat history_ = raw record of conversation saved to Redis. _Chat memory_ = what we choose to inject into context — the basis for reasoning. These are separate.

- **Multi-strategy chat memory** (`HISTORY_STRATEGY`): `full`, `trim`, `summarize`, `rag`, `smart`
- **Smart strategy**: heuristic intent router → dispatches to window, chunk summaries, FAISS vector retrieval, or raw log scan per query
- **Chunk summaries**: messages that leave the sliding window are batched and summarized, then injected as preamble — prevents context drift without compounding compression loss
- **Document RAG**: upload `.txt` files per session; chunks injected into system prompt via FAISS retrieval
- **Dual embedding backend**: `EMBEDDING_PROVIDER=openai` or `local` (HuggingFace)
- **LiteLLM proxy**: switch between OpenAI, Anthropic, and other providers without changing backend code
- **Structured logging**: per-turn NDJSON with cumulative session tokens, routing intent, temperature


## Structure

```
Test_chatbot/
├── backend/
│   ├── app.py                    # FastAPI: /api/chat, /api/documents, LangGraph graph
│   ├── config.py                 # Loads config.yaml + env credentials
│   ├── chat_history/             # Smart strategy components
│   │   ├── clustering.py         # TopicClusterer — LLM-named embedding clusters
│   │   ├── retrieval.py          # retrieve_for_query — dispatches by intent
│   │   ├── router.py             # classify_query — heuristic intent classifier
│   │   ├── store.py              # ChatHistoryStore — per-session FAISS + topic metadata
│   │   └── summarizer.py         # ChunkSummarizer — outside-window batch summaries
│   ├── eval/                     # Evaluation module (importable + CLI)
│   │   ├── dataset.py            # Labeled test set: 18 router cases + sample conversation
│   │   ├── router_eval.py        # Classification accuracy / F1 — sync, no LLM
│   │   ├── summarizer_eval.py    # ROUGE-L + LLM faithfulness judge
│   │   ├── retrieval_eval.py     # Context precision + recall via LLM relevance judge
│   │   ├── e2e_eval.py           # LLM-as-a-judge: correctness, coherence, groundedness
│   │   └── run.py                # CLI runner
│   ├── rag/
│   │   ├── doc_store.py          # DocumentRAGStore — per-session FAISS for uploaded docs
│   │   ├── embeddings.py         # get_embeddings() factory (shared, lru_cache)
│   │   └── store.py              # SessionRAGStore — used by rag strategy
│   ├── helpers/
│   │   └── log.py                # log_message
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── templates/
│   │   └── index.html            # Chat UI: model selector + document upload
│   └── src/
│       └── script.js             # Fetch logic, session ID, doc upload chips
├── nginx/
│   └── nginx.conf                # Proxies /api/ → backend:8000
├── config.yaml                   # Backend application behavior (strategy, eval mode, thresholds)
├── litellm_config.yaml           # Model routing config for LiteLLM
├── logs/
│   └── chat.log                  # Per-message NDJSON
├── models/                       # HuggingFace model cache (local embedding provider)
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
  "usage": {"prompt_tokens": 20, "completion_tokens": 10},
  "strategy": "smart"
}
```

### `POST /api/documents/upload?session_id=<uuid>`
Multipart form upload of a `.txt` file. Chunks and indexes it into the session's FAISS document store. Chunks are injected into the system prompt on every subsequent turn.

### `DELETE /api/documents?session_id=<uuid>`
Clears all uploaded documents for the session.

### `GET /health`
Returns `{"status": "ok"}`.

## Configuration

Two-file split, by intent:

| File | Holds | Per-machine? | In git? |
|---|---|---|---|
| `.env` | Credentials + deployment URLs | Yes | No (gitignored) |
| `config.yaml` | Application behavior (strategy, thresholds, eval mode, paths) | No — same everywhere | Yes |
| `litellm_config.yaml` | LiteLLM proxy: model routing, API key references | No | Yes |

### `.env`

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for GPT models and OpenAI embeddings |
| `ANTHROPIC_API_KEY` | — | Required for Claude models |
| `REDIS_URL` | set by Docker | Redis connection string |
| `LITELLM_BASE_URL` | set by Docker | LiteLLM proxy URL; unset to bypass and call OpenAI directly |
| `EMBEDDING_MODEL_PATH` | `/app/models` | HuggingFace cache path inside the container (volume mount) |

### `config.yaml`

```yaml
llm:
  default_model: gpt-4o-mini
  temperature: 0.0
  system_prompt: "You are a helpful assistant."

embeddings:
  provider: openai                    # openai | local
  model: text-embedding-3-small       # openai provider
  model_name: sentence-transformers/all-MiniLM-L6-v2   # local provider

history:
  strategy: smart                     # full | trim | summarize | rag | smart
  max_tokens: 1000                    # trim
  summarize_after: 3                  # summarize: message count threshold
  rag_top_k: 4                        # rag / smart: top-k retrieved messages
  window_size: 6                      # smart: sliding window
  chunk_size: 6                       # smart: messages per summary chunk
  topic_naming_model: gpt-4o-mini     # smart: cheap LLM for cluster naming
  topic_similarity_threshold: 0.65    # smart: cosine threshold to join existing cluster

rag:
  doc_top_k: 3                        # document chunks injected per query

eval:
  judge_model: gpt-4o-mini            # LLM used for judging

logging:
  chat_log: logs/chat.log
```

Reload behavior: backend reads `config.yaml` at startup. To apply changes, `docker compose restart backend`.

## Adding a new model

1. Add an entry to `litellm_config.yaml`:
```yaml
- model_name: my-model
  litellm_params:
    model: provider/model-id     # e.g. anthropic/claude-sonnet-4-5-20250929
    api_key: os.environ/MY_API_KEY
```
2. Add `<option value="my-model">My Model</option>` to `frontend/templates/index.html`
3. `docker compose restart litellm`

## Chat history

**Terminology:** _Chat history_ = raw log (Redis). _Chat memory_ = what gets injected into context.

### HISTORY_STRATEGY values

| Strategy | What goes to LLM | Cost | Notes |
|---|---|---|---|
| `full` | Entire history every turn | Grows unboundedly | Baseline for comparison |
| `trim` | Last N tokens of history | Flat after threshold | Drops oldest turns silently |
| `summarize` | Rolling summary + last 2 messages | Roughly flat | Compounding compression loss over long sessions |
| `rag` | Top-k semantically similar past messages | Low | Misses recent context |
| `smart` | Router-dispatched (see below) | Adaptive | Most complex, most accurate |

### Smart strategy router

The router classifies each query into an intent and dispatches to the appropriate retrieval path:

```
1. Short query (<4 words) or follow-up signal?
   → CONTINUATION: sliding window only

2. Positional reference ("first/last/Nth question/topic")?
   → POSITION: raw log scan — no embeddings

3. Named topic in query matches a known cluster?
   → TOPIC_RECALL: FAISS filtered by topic metadata

4. Explicit recall phrasing ("earlier you said", "remember when")?
   → SEMANTIC_RECALL: pure FAISS vector search

5. Default
   → NORMAL: sliding window + chunk summaries
```

All paths (except POSITION) also inject chunk summaries — compact LLM-generated summaries of message batches that have left the sliding window. This prevents context drift without compounding compression loss (each batch is summarized once from source, never re-compressed).

**Router evaluation note:** Intent labels for router accuracy testing are _config-independent_ — they represent the semantic intent of the query, not what config values like `WINDOW_SIZE` would make optimal. Router accuracy is a diagnostic metric; end-to-end answer quality (LLM-as-a-judge) is the primary metric. See test prompt set below.

**Router known limitations:**
- CONTINUATION is word-count heuristic (<4 words), not true follow-up detection
- Phrasing like "you listed", "I asked" (implicit recall) not currently detected → routes to NORMAL
- `"we've covered"` contraction bypasses the `we (covered)` recall pattern → routes to NORMAL

**Router TODO:**
- Extend recall patterns for implicit references and contractions
- Replace regex with embedding-based intent matcher or LLM classifier for paraphrase robustness
- Improve retrieval trigger: fire not just on recall signals but also when recent messages are too large or imprecise

## Chat memory
Embeddings are used by both session RAG for memory and document RAG

## Reasoning strategies

Just like chat memory, reasoning has multiple strategies (not yet implemented — see TODO):

- Chain-of-Thought (simplest linear reasoning)
- Self-Consistency
- Tree-of-Thoughts / Graph-of-Thoughts
- Least-to-Most Reasoning
- Reflection / Self-Critique
- Debate / Multi-Agent Reasoning
- Program-of-Thought / Symbolic Reasoning
- ReAct (Reasoning + Acting — popular for agentic systems)

## Testing model switching

The model selector controls **which model answers the next message only**. Conversation history in Redis is model-agnostic. When you switch models mid-conversation:

- The new model receives the full prior context
- Token counts differ because models tokenize differently
- If the new model's context window is smaller than accumulated history, you may get a 502 error

## Testing system prompt changes

1. Edit `llm.system_prompt` in `config.yaml`
2. `docker compose restart backend`
3. Takes effect immediately for new messages in all sessions — the system prompt is prepended at invocation time, not stored in Redis

## Logs

`./logs/` is volume-mounted into the backend container and auto-created on first write.

### `logs/chat.log` — one line per message, NDJSON

```json
{"timestamp": "2026-05-06T22:14:46+00:00", "session_id": "50bbe38e-...", "model": "gpt-4o-mini", "strategy": "smart", "temperature": 0.0, "question": "What was the first question?", "prompt_tokens": 37, "completion_tokens": 10, "session_tokens": 4521, "routing_intent": "position"}
```

| Field | Description |
|---|---|
| `timestamp` | UTC ISO-8601 |
| `session_id` | Browser tab UUID — new UUID = new session |
| `model` | Model name as sent by the frontend |
| `strategy` | Active `history.strategy` |
| `temperature` | Active `llm.temperature` |
| `question` | The user's message |
| `prompt_tokens` | Input tokens for this turn — key metric for strategy comparison |
| `completion_tokens` | Output tokens for this turn |
| `session_tokens` | Cumulative tokens for the session (running total, resets on restart) |
| `routing_intent` | Router decision: `continuation`, `normal`, `position`, `topic_recall`, `semantic_recall` — only present for `smart` strategy |

For verbose request-time inspection, run the backend with `docker compose logs -f backend` — uvicorn writes request/response info to stdout, which Docker captures.

### Useful commands

```bash
# Stream live chat log
tail -f logs/chat.log

# All turns for the smart strategy
jq 'select(.strategy == "smart")' logs/chat.log

# Routing intent breakdown for a session
jq -r 'select(.routing_intent) | [.question, .routing_intent] | @tsv' logs/chat.log

# Final session_tokens per session (last entry per session_id = total cost)
jq -r '[.session_id, .strategy, .session_tokens] | @tsv' logs/chat.log | sort -u -k1,1
```

**Note on `prompt_tokens`:** This is the primary metric for comparing strategies. It shows exactly how much context each strategy injected per turn — the number varies by strategy while `completion_tokens` is roughly constant.


## Test prompt set

Settings used (in `config.yaml`): `history.max_tokens=1000`, `history.summarize_after=3`, `history.rag_top_k=4`, `history.window_size=6`, `history.chunk_size=6`, `history.topic_similarity_threshold=0.65`, `llm.temperature=0.0`

**Expected intent** labels are for supervised router accuracy evaluation. They represent semantic intent of the query, independent of `WINDOW_SIZE` or other config. A mismatch between expected and actual intent is a router bug; whether that bug causes an end-to-end failure depends on whether chunk summaries or the window cover the gap.

| # | Prompt | Expected intent | Assessment |
|---|---|---|---|
| 1 | Who are you? | `continuation` | Baseline: system prompt injection + model persona. Short query (<4 words) heuristic fires correctly. |
| 2 | What is prompt engineering? | `normal` | Topic anchor — referenced by #5, #6, #7. First substantive query, no recall signal. |
| 3 | Is PE important? | `continuation` | Short follow-up. Tests 1-turn memory. |
| 4 | What other skills? | `continuation` | Short follow-up. For `summarize`: SUMMARIZE_AFTER=3 means first summarization fires after turn #2's response (4 messages > 3); by turn #4 two summarization cycles have already run. |
| 5 | PE becoming less important / context engineering? | `normal` | Reasoning test. Self-contained enough that most strategies answer correctly regardless of window — tests reasoning quality, not recall. |
| 6 | Provide PE techniques examples. | `normal` | By turn #6, WINDOW_SIZE=6 excludes turn #2 (PE definition). `full` succeeds; `smart`/`rag` must retrieve. With chunk summaries, `normal` succeeds too. |
| 7 | Combine the skills you listed when I asked 'What other skills?' with the PE techniques you described into a 3-step roadmap. | `semantic_recall` | **Router under-classifies to `normal`** — phrasing "you listed", "I asked", "you described" not in recall patterns. Tests cross-strategy memory: summary fidelity (`summarize`) vs. vector recall (`rag`) vs. chunk summary coverage (`smart`). |
| 8 | What was the first question? | `position` | Positional retrieval. Router correctly matches "first question" pattern. |
| 9 | My favorite language is Python. | `normal` | Fact injection. Binary test: does recall in #10 and #13 succeed? |
| 10 | What is my favorite language? | `semantic_recall` | **Router under-classifies to `normal`** — no recall trigger words. Works while Python info is in window; fails outside it. Semantic label is recall. |
| 11 | What was the first topic we discussed? | `position` | After router fix: "first topic" pattern added. Resolver returns first 2 exchanges (not just first message) so model can infer topic from context. |
| 12 | Summarize what we've covered so far. | `semantic_recall` | **Router under-classifies to `normal`** — "we've covered" contraction bypasses regex. With chunk summaries injected by `normal` path, the answer is now correct despite wrong intent. |
| 13 | Earlier I told you my favorite language — what did you say it was? | `semantic_recall` | "Earlier I" matches recall pattern correctly. Hallucination probe: model must retrieve actual stated language, not fabricate. |
| 14 | What was my last question? | `position` | Tests "last" variant of positional resolver. |
| 15 | What was my 3rd question? | `position` | Tests ordinal variant of positional resolver. |
| 16 | Going back to prompt engineering — give me one more technique. | `topic_recall` | Tests topic-bounded FAISS retrieval. Requires "prompt engineering" to have formed a topic cluster by this point. |
| 17 | Remember when I told you my favorite language? | `semantic_recall` | Tests "remember when" trigger explicitly. |
| 18 | Tell me about the weather today. | `normal` | **Negative test** — irrelevant query with no recall signal. Router must not false-positive into a recall path. |

Router accuracy = correct_intent / 18. Current accuracy in the CLI (no `available_topics` passed): **14/18**. See the **Evaluation > Router known under-classifications** section below for the full failure list.


## Evaluation

The `eval/` module measures the quality of each chat-memory component (router, summarizer, retrieval) and the end-to-end response against a fixed labeled test set in `eval/dataset.py`. All eval runs **inside the backend container** so the environment, LLM config, embeddings, and dependencies match production exactly.

### Mental model: same shape as supervised ML

Standard supervised learning measures a model by comparing predictions to labels:

```
metric = f(y_true, y_pred)
```

Our setup uses the same shape. For the router, `y_true` is a hardcoded intent label in `dataset.py`. For the other components, real ground truth doesn't exist — we synthesize it using a stronger LLM at evaluation time (the "LLM-as-a-judge" pattern).

| Component | `y_pred` | `y_true` (ground truth) | Metric |
|---|---|---|---|
| **Router** | Heuristic router's intent | Hardcoded label in `dataset.py` | Accuracy, per-class F1, confusion matrix |
| **Retrieval** | Top-K retrieved docs | LLM judge marks each doc as relevant/not | Context Precision, Context Recall |
| **Summarizer** | Generated chunk summary | (a) source messages themselves (lexical) (b) LLM judge (semantic) | ROUGE-L, Faithfulness (1-5) |
| **E2E** | The model's response on a labeled prompt | LLM judge scores 1-5 per dimension, against the reference answer | Correctness, Coherence, Groundedness |

### Running it

```bash
# Backend must be running. Then in another terminal:

docker compose exec backend python -m eval.run                  # router (no API calls — pure supervised)
docker compose exec backend python -m eval.run -c summarizer    # ROUGE-L + LLM faithfulness
docker compose exec backend python -m eval.run -c retrieval     # context precision + recall via LLM judge
docker compose exec backend python -m eval.run -c e2e           # LLM-as-a-judge usage instructions
docker compose exec backend python -m eval.run -c all           # router + summarizer + retrieval
```

The CLI prints metrics to stdout. Pipe to a file if you want to save: `... > my_report.txt`.

### Router known under-classifications (current accuracy: 14/18)

Four cases where the heuristic router's prediction differs from the expected intent:

| # | Prompt | Expected | Predicted | Why |
|---|---|---|---|---|
| 7 | "Combine the skills you listed when I asked..." | `semantic_recall` | `normal` | Phrasing "you listed / I asked / you described" not in recall patterns |
| 10 | "What is my favorite language?" | `semantic_recall` | `normal` | No recall trigger words |
| 12 | "Summarize what we've covered so far." | `semantic_recall` | `normal` | `"we've"` contraction bypasses the `we (covered)` regex |
| 16 | "Going back to prompt engineering..." | `topic_recall` | `semantic_recall` | No topic cluster exists in CLI eval (matches `going back to` recall pattern instead). In a live session, `TopicClusterer` would expose "prompt engineering" as a known topic and this case would classify correctly. |

End-to-end answer quality may still be acceptable when chunk summaries cover the gap. The CLI calls `evaluate_router()` without `available_topics` to match a fresh-session baseline; pass `available_topics=[...]` programmatically to simulate a session where topics have already been clustered.

### Programmatic usage

```python
# Router — sync, against hardcoded labels
from eval import evaluate_router
report = evaluate_router()                                 # baseline (no topic cluster)
# report = evaluate_router(available_topics=["prompt engineering"])  # with known topic

# Summarizer — async, generates summaries on a conversation then evaluates
from eval import evaluate_live
report = await evaluate_live(llm, messages, chunk_size=6, window_size=6)

# Retrieval — async, LLM judges per-doc relevance
from eval import evaluate_retrieval
report = await evaluate_retrieval(llm, [
    {"query": "...", "retrieved": [doc1, doc2], "all_docs": full_pool},
])

# E2E — async, LLM-as-a-judge
from eval import evaluate_responses
report = await evaluate_responses(llm, [
    {"question": "...", "response": "...", "reference": "...", "context": "..."},
])
```


## TODO

1. Fix router under-classifications: extend recall patterns for implicit references ("you listed", "I asked") and contractions ("we've covered")
2. Replace router regex with embedding-based intent matcher or LLM classifier
3. Implement `REASONING_STRATEGY` env var (chain-of-thought, reflexion, extended thinking)
4. Graph memory (knowledge graph as a memory layer alongside FAISS)
5. Run E2E eval across all 5 strategies on the 18-prompt test set and compare


## What to experiment with next

- **Streaming responses** — `stream=True` in `ChatOpenAI` + SSE on frontend
- **System prompt editor** — UI textarea to change the prompt without restarting
- **Tool calling** — `@tool` decorated functions + `ToolNode` in LangGraph graph
- **Document summarization on upload** — generate a document-level summary at upload time alongside chunk indexing; route "what is this doc about?" queries to the summary instead of chunks


## Useful links

https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents [Context Engineering]

https://arxiv.org/html/2510.05381v1 [Context Length Alone Hurts LLM Performance Despite Perfect Retrieval]

https://fastpaca.com/blog/llm-memory-systems-explained/ [LLM Memory Systems Explained]

https://www.c-sharpcorner.com/article/how-llm-memory-works-architecture-techniques-and-developer-patterns/ [How LLM Memory Works: Architecture, Techniques, and Developer Patterns]

https://aiagentmemory.org/articles/how-llm-memory-works/ [How LLM Memory Works: Architectures and Mechanisms]

https://arxiv.org/html/2512.20237v1 [MemR3: Memory Retrieval via Reflective Reasoning for LLM Agents]
