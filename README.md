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

- **Multi-strategy chat memory** (`history.strategy`): `full`, `trim`, `summarize`, `rag`, `smart`
- **Smart strategy**: heuristic intent router → dispatches to window, chunk summaries, FAISS vector retrieval, or raw log scan per query
- **Chunk summaries**: messages that leave the sliding window are batched and summarized, then injected as preamble — prevents context drift without compounding compression loss
- **Document RAG**: upload `.txt` files per session; chunks injected into system prompt via FAISS retrieval
- **Dual embedding backend**: `embeddings.provider: openai` or `local` (HuggingFace)
- **LiteLLM proxy**: switch between OpenAI, Anthropic, and other providers without changing backend code
- **Postgres-backed storage**: every chat turn, chunk summary, and eval run persisted to Postgres with full schema (see `initdb/01_schema.sql`) for inspection via any SQL GUI
- **Live-traffic evaluation**: `python -m eval.run --session <uuid>` evaluates a session against four judges (router accuracy, retrieval precision/recall, summarizer faithfulness, e2e quality)


## Structure

```
Test_chatbot/
├── backend/
│   ├── app.py                    # FastAPI: /api/chat, /api/documents, LangGraph graph
│   ├── config.py                 # Loads config.yaml + env credentials
│   ├── db.py                     # Postgres pool + CRUD helpers
│   ├── chat_history/             # Smart strategy components
│   │   ├── clustering.py         # TopicClusterer — LLM-named embedding clusters
│   │   ├── retrieval.py          # retrieve_for_query — dispatches by intent
│   │   ├── router.py             # classify_query — heuristic intent classifier
│   │   ├── store.py              # ChatHistoryStore — per-session FAISS + topic metadata
│   │   └── summarizer.py         # ChunkSummarizer — outside-window batch summaries
│   ├── eval/                     # Evaluation module (CLI + importable evaluators)
│   │   ├── router_eval.py        # Classification accuracy / F1 — sync, no LLM
│   │   ├── summarizer_eval.py    # ROUGE-L + LLM faithfulness judge
│   │   ├── retrieval_eval.py     # Context precision + recall via LLM relevance judge
│   │   ├── e2e_eval.py           # LLM-as-a-judge: correctness, coherence, groundedness
│   │   └── run.py                # CLI: --session <uuid> reads from Postgres
│   ├── rag/
│   │   ├── doc_store.py          # DocumentRAGStore — per-session FAISS for uploaded docs
│   │   ├── embeddings.py         # get_embeddings() factory (shared, lru_cache)
│   │   └── store.py              # SessionRAGStore — used by rag strategy
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── templates/
│   │   └── index.html            # Chat UI: model selector + document upload
│   └── src/
│       └── script.js             # Fetch logic, session ID, doc upload chips
├── nginx/
│   └── nginx.conf                # Proxies /api/ → backend:8000
├── initdb/
│   └── 01_schema.sql             # Postgres schema (loaded by docker-entrypoint on first init)
├── config.yaml                   # Backend application behavior (strategy, thresholds, judge model)
├── litellm_config.yaml           # Model routing config for LiteLLM
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
# Start Redis (session memory) and Postgres (chat + eval data)
docker compose up redis postgres -d

# Backend
cd backend
pip install -r requirements.txt
REDIS_URL=redis://localhost:6379 \
POSTGRES_DSN=postgresql://chatbot:changeme@localhost:5432/chatbot \
  uvicorn app:app --reload --port 8000

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
| `PG_USER` | `chatbot` | Postgres username (used by both the postgres service and the backend DSN) |
| `PG_PASSWORD` | `changeme` | Postgres password |
| `PG_DB` | `chatbot` | Postgres database name |

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

## Data storage

All chat traffic and evaluation results live in PostgreSQL (`postgres` service in docker-compose). Connect with any local GUI — pgAdmin, DBeaver, TablePlus, or `psql`:

```bash
# psql, from your host
psql postgresql://chatbot:changeme@localhost:5432/chatbot

# or from inside the postgres container
docker compose exec postgres psql -U chatbot -d chatbot
```

The schema lives in [`initdb/01_schema.sql`](initdb/01_schema.sql) and is loaded on first DB init. Four tables:

| Table | One row per | Purpose |
|---|---|---|
| `sessions` | chat session (UUID) | strategy, temperature, system prompt — constants for the session |
| `turns` | chat message | question, response, tokens, routing decision, retrieved docs, context injected, and user-overridable ground truth (`intent_user_override`, `reference_answer`) |
| `chunk_summaries` | summarizer-produced chunk | source messages + generated summary — only populated for `smart` strategy |
| `eval_runs` | evaluation invocation | aggregate metrics + per-turn judge scores; one row per `eval.run --session` call |

### Verbose container logs

For request-time inspection, run `docker compose logs -f backend` — uvicorn writes request/response info to stdout, which Docker captures.

### Useful SQL

```sql
-- All sessions, most recent first
SELECT session_id, created_at, strategy, temperature FROM sessions ORDER BY created_at DESC;

-- All turns for a session
SELECT turn_number, model, question, response, routing->>'intent' AS intent
FROM turns WHERE session_id = '50bbe38e-...' ORDER BY turn_number;

-- Token totals per session (matches what chat.log used to show)
SELECT session_id,
       MAX((tokens->>'session_total')::int) AS total_tokens,
       COUNT(*) AS turns
FROM turns GROUP BY session_id;

-- Routing intent breakdown for one session
SELECT turn_number, question, routing->>'intent' AS intent FROM turns
WHERE session_id = '50bbe38e-...' ORDER BY turn_number;

-- Most recent eval run for a session
SELECT evaluated_at, judge_model, aggregate_metrics
FROM eval_runs WHERE session_id = '50bbe38e-...'
ORDER BY evaluated_at DESC LIMIT 1;
```

### Manual labels and references

Two columns on `turns` accept user-provided ground truth (durable across eval runs):

```sql
-- Override an LLM-judge intent label (takes precedence over judge_model output)
UPDATE turns SET intent_user_override = 'semantic_recall'
WHERE session_id = '50bbe38e-...' AND turn_number = 7;

-- Set a gold reference answer for E2E correctness judging
UPDATE turns SET reference_answer = 'The first topic was prompt engineering.'
WHERE session_id = '50bbe38e-...' AND turn_number = 11;
```

Re-run `eval.run --session <uuid>` after editing — the overrides take effect immediately without re-querying the LLM judge.


## Test prompt set

## Evaluation

Evaluation measures the quality of each chat-memory component (router, summarizer, retrieval) and the end-to-end response on **live chat traffic** stored in Postgres. The CLI is invoked with a `session_id` and reads everything it needs from the `turns`, `chunk_summaries`, and (after the run) `eval_runs` tables.

There is no hardcoded benchmark dataset — the project's goal is collaborative improvement of evaluation, and a maintainer-curated baseline would bias the system toward author assumptions. If you want a reproducible baseline, follow the [Suggested baseline prompts](#suggested-baseline-prompts-optional) below; they are *suggestions*, not requirements.

### Mental model: same shape as supervised ML

Standard supervised learning measures a model by comparing predictions to labels:

```
metric = f(y_true, y_pred)
```

Our setup uses the same shape. For the router, real users ask arbitrary queries — there are no pre-labeled `y_true` values. So we synthesize ground truth using a stronger LLM at evaluation time ("LLM-as-a-judge"). For other components there's no fixed ground truth either; the judge provides it.

| Component | `y_pred` (recorded during chat) | `y_true` (ground truth at eval time) | Metric |
|---|---|---|---|
| **Router** | Heuristic router's intent (`turns.routing.intent`) | LLM judge labels each query, falling back to `turns.intent_user_override` if the user has set one | Accuracy, per-class F1, confusion matrix |
| **Retrieval** | Top-K retrieved docs (`turns.retrieved`) | LLM judge marks each doc as relevant/not | Context Precision, Context Recall |
| **Summarizer** | Generated chunk summary (`chunk_summaries.summary_text`) | (a) source messages themselves (lexical) (b) LLM judge (semantic) | ROUGE-L, Faithfulness (1-5) |
| **E2E** | The model's response (`turns.response`) | LLM judge scores 1-5 per dimension, using `turns.reference_answer` if set | Correctness, Coherence, Groundedness |

### Running it

The backend must be up (`docker compose up`). In another terminal, with a session UUID from the browser:

```bash
docker compose exec backend python -m eval.run --session <uuid>                  # all components
docker compose exec backend python -m eval.run --session <uuid> -c router
docker compose exec backend python -m eval.run --session <uuid> -c retrieval
docker compose exec backend python -m eval.run --session <uuid> -c summarizer
docker compose exec backend python -m eval.run --session <uuid> -c e2e
```

Each invocation:
1. Fetches the session's turns + chunk summaries from Postgres
2. For router eval: LLM-judges each unlabeled turn's intent (cached back to `turns.intent_judge_label` so re-runs don't re-pay)
3. Runs the appropriate judges (retrieval relevance, summarizer faithfulness, e2e correctness/coherence/groundedness)
4. Prints aggregate metrics + per-turn breakdown to stdout
5. Appends an `eval_runs` row for historical comparison

### Suggested baseline prompts (optional)

A set of 18 prompts I (the author) have been using to probe each chat-memory pattern. They're suggestions, not part of the codebase — you (or another contributor) can refine them, replace them, or design your own. If you want a reproducible baseline across changes, type these into the chat UI in order, then run `eval.run --session <uuid>` against the resulting session.

After chatting, optionally set `intent_user_override` via SQL on any turn where you disagree with the LLM judge's intent label (see [Manual labels and references](#manual-labels-and-references)).

| # | Prompt | Suggested intent | Notes |
|---|---|---|---|
| 1 | Who are you? | `continuation` | Short query (<4 words) heuristic. |
| 2 | What is prompt engineering? | `normal` | First substantive query; no recall signal. |
| 3 | Is PE important? | `continuation` | Short follow-up. |
| 4 | What other skills? | `continuation` | Short follow-up. |
| 5 | Is PE becoming less important as context engineering matures? | `normal` | Reasoning test; self-contained. |
| 6 | Provide PE techniques examples. | `normal` | By turn #6, `window_size=6` excludes turn #2 — `full` succeeds, `smart`/`rag` must retrieve. |
| 7 | Combine the skills you listed when I asked 'What other skills?' with the PE techniques you described into a 3-step roadmap. | `semantic_recall` | Router currently under-classifies to `normal` — phrasing not in recall patterns. |
| 8 | What was the first question? | `position` | Tests positional retrieval. |
| 9 | My favorite language is Python. | `normal` | Fact injection — recall test in #10, #13. |
| 10 | What is my favorite language? | `semantic_recall` | Router currently under-classifies to `normal` — no recall trigger words. |
| 11 | What was the first topic we discussed? | `position` | Tests "first topic" pattern. |
| 12 | Summarize what we've covered so far. | `semantic_recall` | Router currently under-classifies — `"we've"` contraction bypasses regex. |
| 13 | Earlier I told you my favorite language — what did you say it was? | `semantic_recall` | Hallucination probe; recall pattern matches correctly. |
| 14 | What was my last question? | `position` | Tests "last" variant of positional resolver. |
| 15 | What was my 3rd question? | `position` | Tests ordinal variant. |
| 16 | Going back to prompt engineering — give me one more technique. | `topic_recall` | Requires "prompt engineering" to have formed a topic cluster. |
| 17 | Remember when I told you my favorite language? | `semantic_recall` | Tests "remember when" trigger explicitly. |
| 18 | Tell me about the weather today. | `normal` | Negative test — router must not false-positive into a recall path. |

After running these and triggering `eval.run --session <uuid>`, you'd expect the router to under-classify intents at turns #7, #10, #12 (known limitations — see TODO).

### Programmatic usage

The component evaluators are pure functions and can be called from your own scripts. The CLI in `eval/run.py` is one consumer; you can build others (e.g., a cross-strategy comparison script).

```python
from eval import evaluate_router, evaluate_retrieval, evaluate_responses, evaluate_summarizer
from eval.router_eval import RouterCase
from chat_history.router import Intent

# Router — sync, requires labeled cases
cases = [RouterCase(id=1, prompt="What is X?", expected_intent=Intent.NORMAL)]
report = evaluate_router(cases=cases)

# Summarizer — async, takes (source_messages, generated_summary) pairs
report = await evaluate_summarizer(llm, [(messages, summary_text), ...])

# Retrieval — async, LLM judges per-doc relevance
report = await evaluate_retrieval(llm, [
    {"query": "...", "retrieved": [doc1, doc2], "all_docs": full_pool},
])

# E2E — async, LLM-as-a-judge
report = await evaluate_responses(llm, [
    {"question": "...", "response": "...", "reference": "...", "context": "..."},
])
```


## TODO

1. Fix router under-classifications: extend recall patterns for implicit references ("you listed", "I asked") and contractions ("we've covered")
2. Replace router regex with embedding-based intent matcher or LLM classifier
3. Implement reasoning strategy variants (chain-of-thought, reflexion, extended thinking)
4. Graph memory (knowledge graph as a memory layer alongside FAISS)
5. Cross-strategy comparison script that runs the suggested baseline prompts against each `history.strategy` and compares the resulting `eval_runs` rows


## What to experiment with next

- **Tool calling** — `@tool` decorated functions + `ToolNode` in LangGraph graph
- **Reasoning logic** -- implement resoning logic and metrics that would help evaluation of it


## Useful links

https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents [Context Engineering]

https://arxiv.org/html/2510.05381v1 [Context Length Alone Hurts LLM Performance Despite Perfect Retrieval]

https://fastpaca.com/blog/llm-memory-systems-explained/ [LLM Memory Systems Explained]

https://www.c-sharpcorner.com/article/how-llm-memory-works-architecture-techniques-and-developer-patterns/ [How LLM Memory Works: Architecture, Techniques, and Developer Patterns]

https://aiagentmemory.org/articles/how-llm-memory-works/ [How LLM Memory Works: Architectures and Mechanisms]

https://arxiv.org/html/2512.20237v1 [MemR3: Memory Retrieval via Reflective Reasoning for LLM Agents]
