import json
import os
from datetime import datetime, timezone
from pathlib import Path

_DEBUG = os.getenv("DEBUG", "false").lower() == "true"

CHAT_LOG = Path(os.getenv("CHAT_LOG", "logs/chat.log"))
DEBUG_LOG = Path(os.getenv("DEBUG_LOG", "logs/debug.log"))

_session_tokens: dict[str, int] = {}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def log_message(
    session_id: str,
    model: str,
    question: str,
    usage: dict,
    strategy: str = "full",
    temperature: float = 1.0,
    routing_intent: str = "",
) -> None:
    """Append one line to chat.log after every successful chat response."""
    turn_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    _session_tokens[session_id] = _session_tokens.get(session_id, 0) + turn_tokens

    payload: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "model": model,
        "strategy": strategy,
        "temperature": temperature,
        "question": question,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "session_tokens": _session_tokens[session_id],
    }
    if routing_intent:
        payload["routing_intent"] = routing_intent
    _write(CHAT_LOG, payload)


def log_debug(location: str, message: str, data: dict) -> None:
    """Append one line to debug.log. Only active when DEBUG=true."""
    if not _DEBUG:
        return
    _write(DEBUG_LOG, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": location,
        "message": message,
        "data": data,
    })
