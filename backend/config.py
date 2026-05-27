"""Application config loader.

Two sources of configuration, by intent:

  config.yaml  — application behavior (history strategy, eval mode, thresholds).
                 Same on every machine; version-controlled.

  environment  — credentials and deployment URLs. Per-machine; loaded from .env
                 (docker-compose for containers, python-dotenv for local dev).
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Credentials and deployment URLs                                              #
# --------------------------------------------------------------------------- #

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
LITELLM_BASE_URL: str | None = os.getenv("LITELLM_BASE_URL") or None
EMBEDDING_MODEL_PATH: str = os.getenv("EMBEDDING_MODEL_PATH", "/app/models")


# --------------------------------------------------------------------------- #
# Application config (config.yaml)                                             #
# --------------------------------------------------------------------------- #

def _find_config() -> Path:
    explicit = os.getenv("CONFIG_PATH")
    if explicit:
        return Path(explicit)
    for candidate in (Path("config.yaml"), Path("../config.yaml"), Path("/app/config.yaml")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "config.yaml not found. Set CONFIG_PATH env var or place config.yaml "
        "in CWD, parent dir, or /app/config.yaml."
    )


CONFIG: dict = yaml.safe_load(_find_config().read_text())
