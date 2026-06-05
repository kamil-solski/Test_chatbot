"""Postgres access layer.

Connection pool managed by app.py lifespan. All eval data (per-turn chat records,
chunk summaries, evaluation runs) is stored here. See initdb/01_schema.sql for the
schema.
"""
import json
from typing import Any

import asyncpg

from config import POSTGRES_DSN

_pool: asyncpg.Pool | None = None


# --------------------------------------------------------------------------- #
# Lifecycle                                                                     #
# --------------------------------------------------------------------------- #

async def init_pool() -> asyncpg.Pool:
    """Initialize the global connection pool. Idempotent."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _require_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("db pool not initialized; call init_pool() first")
    return _pool


# --------------------------------------------------------------------------- #
# Sessions                                                                      #
# --------------------------------------------------------------------------- #

async def ensure_session(
    session_id: str,
    strategy: str,
    temperature: float,
    system_prompt: str,
) -> None:
    """Insert a session row if it doesn't already exist. No-op on subsequent calls."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (session_id, strategy, temperature, system_prompt)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id, strategy, temperature, system_prompt,
        )


# --------------------------------------------------------------------------- #
# Turns                                                                         #
# --------------------------------------------------------------------------- #

async def next_turn_number(session_id: str) -> int:
    """Return the next turn_number to use for this session (1-indexed)."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COALESCE(MAX(turn_number), 0) + 1 AS n FROM turns WHERE session_id = $1",
            session_id,
        )
        return int(row["n"])


async def insert_turn(
    session_id: str,
    turn_number: int,
    model: str,
    question: str,
    response: str,
    context_injected: str | None,
    routing: dict | None,
    retrieved: list[dict] | None,
    tokens: dict,
) -> int:
    """Insert one turn row. Returns the row id."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO turns (
                session_id, turn_number, model, question, response,
                context_injected, routing, retrieved, tokens
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            session_id, turn_number, model, question, response,
            context_injected,
            json.dumps(routing) if routing is not None else None,
            json.dumps(retrieved) if retrieved is not None else None,
            json.dumps(tokens),
        )
        return int(row["id"])


async def fetch_session_turns(session_id: str) -> list[dict]:
    """Return all turns for a session, ordered by turn_number."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, turn_number, ts, model, question, response, context_injected,
                   routing, retrieved, tokens,
                   intent_user_override, reference_answer, intent_judge_label
            FROM turns
            WHERE session_id = $1
            ORDER BY turn_number
            """,
            session_id,
        )
        return [_decode_turn_row(r) for r in rows]


async def update_turn_judge_label(turn_id: int, intent_judge_label: str) -> None:
    """Cache an LLM-judge intent label so we don't re-query on subsequent eval runs."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE turns SET intent_judge_label = $1 WHERE id = $2",
            intent_judge_label, turn_id,
        )


def _decode_turn_row(r: asyncpg.Record) -> dict:
    """Convert an asyncpg Record to a plain dict, parsing JSONB fields."""
    d = dict(r)
    for k in ("routing", "retrieved", "tokens"):
        v = d.get(k)
        if isinstance(v, str):
            d[k] = json.loads(v)
    return d


# --------------------------------------------------------------------------- #
# Chunk summaries                                                               #
# --------------------------------------------------------------------------- #

async def insert_chunk_summary(
    session_id: str,
    chunk_index: int,
    source_messages: list[dict],
    summary_text: str,
) -> None:
    pool = _require_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chunk_summaries (session_id, chunk_index, source_messages, summary_text)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (session_id, chunk_index) DO NOTHING
            """,
            session_id, chunk_index, json.dumps(source_messages), summary_text,
        )


async def fetch_session_chunks(session_id: str) -> list[dict]:
    pool = _require_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT chunk_index, ts, source_messages, summary_text
            FROM chunk_summaries
            WHERE session_id = $1
            ORDER BY chunk_index
            """,
            session_id,
        )
        result = []
        for r in rows:
            d = dict(r)
            if isinstance(d["source_messages"], str):
                d["source_messages"] = json.loads(d["source_messages"])
            result.append(d)
        return result


# --------------------------------------------------------------------------- #
# Eval runs                                                                     #
# --------------------------------------------------------------------------- #

async def insert_eval_run(
    session_id: str,
    judge_model: str,
    turns_evaluated: int,
    aggregate_metrics: dict[str, Any],
    per_turn_results: list[dict[str, Any]],
) -> int:
    pool = _require_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO eval_runs (
                session_id, judge_model, turns_evaluated,
                aggregate_metrics, per_turn_results
            )
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            session_id, judge_model, turns_evaluated,
            json.dumps(aggregate_metrics), json.dumps(per_turn_results),
        )
        return int(row["id"])
