"""Env-driven config. Loads .env if present (no python-dotenv dep)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 4096


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ if not already set."""
    if path is None:
        path = Path.cwd() / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings() -> Settings:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (env or .env)")
    return Settings(
        anthropic_api_key=api_key,
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
    )
