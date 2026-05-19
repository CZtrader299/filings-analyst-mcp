"""Configuration constants and env-var-driven defaults.

SEC EDGAR requires every client to identify itself with a User-Agent that
includes a real contact email. The default below uses Dan Krawczun's
research email; override via the ``FILINGS_ANALYST_USER_AGENT`` env var if
you fork this project.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- SEC EDGAR ---------------------------------------------------------------

# SEC's fair-access policy: identify yourself with a real email and stay
# under 10 requests/second. See https://www.sec.gov/os/accessing-edgar-data.
USER_AGENT = os.environ.get(
    "FILINGS_ANALYST_USER_AGENT",
    "Dan Krawczun research@krawczun.com",
)

# Cache directory for downloaded filings. Override with FILINGS_ANALYST_CACHE_DIR.
CACHE_DIR = Path(
    os.environ.get(
        "FILINGS_ANALYST_CACHE_DIR",
        str(Path.home() / ".filings_analyst_cache"),
    )
)

# Five large-cap names spanning tech (AAPL, MSFT), banking (JPM, BAC), and
# energy (XOM). Small enough to ingest quickly, diverse enough to exercise
# section-extraction edge cases (banks file very different MD&A sections
# than tech firms).
STARTER_TICKERS = ["AAPL", "MSFT", "JPM", "BAC", "XOM"]

# --- LLM provider routing ----------------------------------------------------

# One of: none, auto, claude_cli, anthropic_api, openai_api.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto").strip().lower()

# One of: local, openai.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local").strip().lower()

# --- Model defaults (env-overridable) ---------------------------------------

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
LOCAL_EMBEDDING_MODEL = os.environ.get(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Network / timeouts ------------------------------------------------------

REQUEST_TIMEOUT = int(os.environ.get("FILINGS_ANALYST_TIMEOUT", "30"))
LLM_TIMEOUT = int(os.environ.get("FILINGS_ANALYST_LLM_TIMEOUT", "120"))

# API keys (read-only — providers re-read these from os.environ at call time
# so tests can monkeypatch without reloading this module).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
