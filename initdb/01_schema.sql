-- Test Chatbot — Postgres schema
-- Loaded on first container start by docker-entrypoint-initdb.d.
-- Manual changes after first start require `docker compose down -v` + `up` to re-init.

-- TODO: later as project would grow replace this initdb with django ORM

-- Session-level metadata (constant across turns within a session)
CREATE TABLE IF NOT EXISTS sessions (
    session_id      UUID PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    strategy        TEXT NOT NULL,             -- full | trim | summarize | rag | smart
    temperature     REAL NOT NULL,
    system_prompt   TEXT NOT NULL
);

-- One row per chat turn
CREATE TABLE IF NOT EXISTS turns (
    id                       SERIAL PRIMARY KEY,
    session_id               UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    turn_number              INTEGER NOT NULL,
    ts                       TIMESTAMPTZ NOT NULL DEFAULT now(),
    model                    TEXT NOT NULL,    -- can vary per turn (user selects)

    question                 TEXT NOT NULL,
    response                 TEXT NOT NULL,
    context_injected         TEXT,             -- full system prompt + extras actually sent to LLM

    routing                  JSONB,            -- {intent, reason, topic_filter}; null for non-smart strategies
    retrieved                JSONB,            -- [{role, content}, ...]; null when strategy doesn't retrieve
    tokens                   JSONB NOT NULL,   -- {prompt, completion, session_total}

    -- User-provided ground truth (durable across eval runs)
    intent_user_override     TEXT,             -- e.g. "semantic_recall" — overrides judge label
    reference_answer         TEXT,             -- gold E2E answer for this turn

    -- Cached LLM judge output (populated by eval.run, avoids re-querying)
    intent_judge_label       TEXT,

    UNIQUE (session_id, turn_number)
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_intent ON turns((routing->>'intent'));

-- One row per chunk summary produced by ChunkSummarizer (smart strategy only)
CREATE TABLE IF NOT EXISTS chunk_summaries (
    id              SERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_messages JSONB NOT NULL,            -- [{role, content}, ...]
    summary_text    TEXT NOT NULL,
    UNIQUE (session_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunk_summaries(session_id);

-- One row per evaluation invocation (eval.run --session <uuid>)
CREATE TABLE IF NOT EXISTS eval_runs (
    id                  SERIAL PRIMARY KEY,
    session_id          UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    evaluated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    judge_model         TEXT NOT NULL,
    turns_evaluated     INTEGER NOT NULL,
    aggregate_metrics   JSONB NOT NULL,        -- {router_accuracy, retrieval_precision_mean, e2e_correctness_mean, ...}
    per_turn_results    JSONB NOT NULL         -- [{turn_number, intent_judge_label, e2e_scores, retrieval_metrics, ...}]
);
CREATE INDEX IF NOT EXISTS idx_evals_session ON eval_runs(session_id);
