"""LLM provider routing.

Mirrors the pattern from Dan's LSE Buyback Scraper ``ai_reviewer.py``, but
simpler: we only need free-form text generation here, not JSON-schema
constrained extraction. If a downstream caller (e.g. a future eval-time
grader) needs structured output, extend this then rather than now.

Providers (priority order for ``"auto"``):

1. ``anthropic_api`` — Messages API; needs ``ANTHROPIC_API_KEY``.
2. ``openai_api`` — Chat Completions API; needs ``OPENAI_API_KEY``.
3. ``claude_cli`` — local ``claude -p``; covered by Max-plan Agent SDK
   credit (activates 2026-06-15).

``"none"`` disables generation entirely; ``generate()`` returns ``None``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

import requests

from . import config


PROVIDERS = {"none", "auto", "claude_cli", "anthropic_api", "openai_api"}


class LLMProvider:
    """Resolve and call a configured LLM backend."""

    def __init__(self, provider: Optional[str] = None):
        requested = (provider or config.LLM_PROVIDER or "none").strip().lower()
        if requested == "claude":
            requested = "claude_cli"
        if requested not in PROVIDERS:
            print(f"  Warning: unknown LLM provider {requested!r}, disabling generation")
            requested = "none"

        self.requested = requested
        self.provider = self._resolve_provider(requested)
        self.available = self.provider != "none"

    # --- Resolution ------------------------------------------------------

    def _resolve_provider(self, requested: str) -> str:
        if requested == "none":
            return "none"
        if requested == "anthropic_api":
            return "anthropic_api" if self._check_anthropic_api_available() else "none"
        if requested == "openai_api":
            return "openai_api" if self._check_openai_api_available() else "none"
        if requested == "claude_cli":
            return "claude_cli" if self._check_cli_available() else "none"
        if requested == "auto":
            # Hosted APIs first (fastest, highest quality), then the local
            # CLI as the always-available fallback for users with a Claude
            # subscription but no API key set.
            if self._check_anthropic_api_available():
                return "anthropic_api"
            if self._check_openai_api_available():
                return "openai_api"
            if self._check_cli_available():
                return "claude_cli"
        return "none"

    @staticmethod
    def _check_anthropic_api_available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    @staticmethod
    def _check_openai_api_available() -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    @staticmethod
    def _check_cli_available() -> bool:
        # When Claude Code itself is the parent process, it sets CLAUDECODE=1
        # and a nested ``claude`` call would re-enter the agent harness.
        # Pop it before probing so the spawned shell behaves like a normal
        # interactive user.
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        # shutil.which honors PATHEXT on Windows, finding claude.cmd / .exe.
        return shutil.which("claude") is not None

    @staticmethod
    def _clean_env() -> dict[str, str]:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        return env

    # --- Public API ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> Optional[str]:
        """Generate text. Returns ``None`` on any failure."""
        if not self.available:
            return None
        try:
            if self.provider == "claude_cli":
                return self._run_claude_cli(prompt, system)
            if self.provider == "anthropic_api":
                return self._run_anthropic_api(prompt, max_tokens, system)
            if self.provider == "openai_api":
                return self._run_openai_api(prompt, max_tokens, system)
        except TimeoutError:
            print(f"  Warning: LLM provider {self.provider} timed out")
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: LLM provider {self.provider} failed: {exc}")
            return None
        return None

    # --- Provider dispatch ----------------------------------------------

    def _run_claude_cli(self, prompt: str, system: Optional[str]) -> Optional[str]:
        claude_path = shutil.which("claude") or "claude"
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        # SEC filings contain typographic characters (curly quotes,
        # non-breaking hyphens, en/em dashes) that crash Windows'
        # default cp1252 stdin encoding when piped to a subprocess.
        # Encode to UTF-8 bytes explicitly and pass via the bytes
        # interface so Python doesn't try to apply the console codepage.
        encoded = full_prompt.encode("utf-8", errors="replace")
        try:
            result = subprocess.run(
                [
                    claude_path,
                    "-p",
                    "-",
                    "--output-format",
                    "text",
                ],
                input=encoded,
                capture_output=True,
                timeout=config.LLM_TIMEOUT,
                env=self._clean_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError from exc
        # Tests mock subprocess.run with text results; production passes
        # bytes through since we don't pass text=True. Handle both.
        def _to_text(blob: object) -> str:
            if blob is None:
                return ""
            if isinstance(blob, bytes):
                return blob.decode("utf-8", errors="replace")
            return str(blob)

        if result.returncode != 0:
            stderr_text = _to_text(result.stderr)
            print(f"  Warning: Claude CLI exited {result.returncode}: {stderr_text[:200]}")
            return None
        stdout_text = _to_text(result.stdout)
        return stdout_text.strip() or None

    def _run_anthropic_api(
        self, prompt: str, max_tokens: int, system: Optional[str]
    ) -> Optional[str]:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = os.environ.get("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL)
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=body,
                headers=headers,
                timeout=config.LLM_TIMEOUT,
            )
        except requests.Timeout as exc:
            raise TimeoutError from exc
        if not response.ok:
            print(f"  Warning: Anthropic API HTTP {response.status_code}: {response.text[:200]}")
            return None
        payload = response.json()
        try:
            return payload["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return None

    def _run_openai_api(
        self, prompt: str, max_tokens: int, system: Optional[str]
    ) -> Optional[str]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", config.OPENAI_MODEL)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=config.LLM_TIMEOUT,
            )
        except requests.Timeout as exc:
            raise TimeoutError from exc
        if not response.ok:
            print(f"  Warning: OpenAI API HTTP {response.status_code}: {response.text[:200]}")
            return None
        payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
