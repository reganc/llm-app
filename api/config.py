"""Centralized config: env vars + persisted runtime settings."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

log = logging.getLogger("llm-api")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
CHROMA_DIR = DATA_DIR / "chroma"
CONV_DIR = DATA_DIR / "conversations"
FINETUNE_DIR = DATA_DIR / "finetune"
SETTINGS_FILE = DATA_DIR / "settings.json"
AUX_SITES_FILE = DATA_DIR / "auxiliary_sites.json"
STATIC_DIR = Path(__file__).parent / "static"

for d in (DATA_DIR, CHROMA_DIR, CONV_DIR, FINETUNE_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    ollama_url: str
    default_model: str
    embed_model: str
    api_key: str
    max_tokens: int
    cors_origins: list[str]

    memory_enabled: bool
    chunk_size: int
    chunk_overlap: int
    memory_top_k: int

    search_enabled: bool
    search_always: bool
    searxng_url: str
    auto_search_threshold: float
    search_results: int
    search_fetch_timeout: int

    firecrawl_url: str
    firecrawl_api_key: str

    x_search_enabled: bool
    x_cli_path: str
    x_search_count: int

    classifier_timeout: float
    context_char_limit: int
    max_inject_chars: int


def _csv(name: str, default: str) -> list[str]:
    return [s.strip() for s in os.getenv(name, default).split(",") if s.strip()]


CFG = Config(
    ollama_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/"),
    default_model=os.getenv("DEFAULT_MODEL", "mistral:7b-instruct-q4_K_M"),
    embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    api_key=os.getenv("API_KEY", "change-me-in-production"),
    max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
    cors_origins=_csv("CORS_ORIGINS", "*"),
    memory_enabled=os.getenv("MEMORY_ENABLED", "true").lower() == "true",
    chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
    memory_top_k=int(os.getenv("MEMORY_TOP_K", "5")),
    search_enabled=os.getenv("SEARCH_ENABLED", "true").lower() == "true",
    search_always=os.getenv("SEARCH_ALWAYS", "false").lower() == "true",
    searxng_url=os.getenv("SEARXNG_URL", "http://searxng:8080").rstrip("/"),
    auto_search_threshold=float(os.getenv("AUTO_SEARCH_THRESHOLD", "0.35")),
    search_results=int(os.getenv("SEARCH_RESULTS", "5")),
    search_fetch_timeout=int(os.getenv("SEARCH_FETCH_TIMEOUT", "15")),
    firecrawl_url=os.getenv("FIRECRAWL_URL", "").rstrip("/"),
    firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY", "fc-local"),
    x_search_enabled=os.getenv("X_SEARCH_ENABLED", "false").lower() == "true",
    x_cli_path=os.getenv("X_CLI_PATH", "/usr/local/bin/x-cli"),
    x_search_count=int(os.getenv("X_SEARCH_COUNT", "10")),
    classifier_timeout=float(os.getenv("CLASSIFIER_TIMEOUT", "4.0")),
    context_char_limit=int(os.getenv("CONTEXT_CHAR_LIMIT", "12000")),
    max_inject_chars=int(os.getenv("MAX_INJECT_CHARS", "5000")),
)

# ── Mutable runtime settings (persisted to disk) ─────────────────────────────
_lock = Lock()


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_settings: dict = _load_settings()


def get_setting(key: str, default=None):
    with _lock:
        return _settings.get(key, default)


def set_setting(key: str, value) -> None:
    with _lock:
        _settings[key] = value
        SETTINGS_FILE.write_text(json.dumps(_settings, indent=2))


def get_active_model() -> str:
    """Currently selected default model (settings overrides env)."""
    return get_setting("default_model", CFG.default_model)


def get_default_system_prompt_key() -> str:
    return get_setting("default_system_prompt_key", os.getenv("DEFAULT_SYSTEM_PROMPT", "default"))
