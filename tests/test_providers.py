"""Tests for the LLM provider router.

All subprocess and HTTP calls are mocked. No network, no real ``claude``
binary.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from filings_analyst import providers


# --- Provider resolution ----------------------------------------------------


def test_unknown_provider_falls_back_to_none(capsys):
    p = providers.LLMProvider(provider="banana")
    assert p.provider == "none"
    assert p.available is False


def test_resolve_none_explicit():
    p = providers.LLMProvider(provider="none")
    assert p.provider == "none"
    assert p.available is False


def test_resolve_anthropic_when_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    p = providers.LLMProvider(provider="anthropic_api")
    assert p.provider == "anthropic_api"
    assert p.available is True


def test_resolve_anthropic_disabled_when_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = providers.LLMProvider(provider="anthropic_api")
    assert p.provider == "none"


def test_resolve_openai_when_key_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    p = providers.LLMProvider(provider="openai_api")
    assert p.provider == "openai_api"


def test_resolve_auto_prefers_anthropic_then_openai_then_cli(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    p = providers.LLMProvider(provider="auto")
    assert p.provider == "anthropic_api"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    p2 = providers.LLMProvider(provider="auto")
    assert p2.provider == "openai_api"

    monkeypatch.delenv("OPENAI_API_KEY")
    with patch("filings_analyst.providers.shutil.which", return_value="/usr/bin/claude"):
        p3 = providers.LLMProvider(provider="auto")
        assert p3.provider == "claude_cli"

    with patch("filings_analyst.providers.shutil.which", return_value=None):
        p4 = providers.LLMProvider(provider="auto")
        assert p4.provider == "none"


def test_check_cli_available_uses_shutil_which(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")  # should not affect resolution
    with patch("filings_analyst.providers.shutil.which", return_value="C:\\bin\\claude.cmd") as mock_which:
        assert providers.LLMProvider._check_cli_available() is True
        mock_which.assert_called_with("claude")
    with patch("filings_analyst.providers.shutil.which", return_value=None):
        assert providers.LLMProvider._check_cli_available() is False


# --- Generation dispatch ----------------------------------------------------


def test_generate_returns_none_when_unavailable():
    p = providers.LLMProvider(provider="none")
    assert p.generate("hello") is None


def test_claude_cli_invocation(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    fake_result = MagicMock(returncode=0, stdout="cli reply\n", stderr="")
    with patch("filings_analyst.providers.shutil.which", return_value="/usr/bin/claude"), patch(
        "filings_analyst.providers.subprocess.run", return_value=fake_result
    ) as mock_run:
        p = providers.LLMProvider(provider="claude_cli")
        assert p.provider == "claude_cli"
        out = p.generate("hi", system="be brief")
        assert out == "cli reply"
        # Verify CLAUDECODE was stripped from the env passed to subprocess.
        called_env = mock_run.call_args.kwargs["env"]
        assert "CLAUDECODE" not in called_env
        # Verify -p mode used.
        cmd = mock_run.call_args.args[0]
        assert cmd[1] == "-p"


def test_claude_cli_nonzero_returncode_returns_none():
    fake = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch("filings_analyst.providers.shutil.which", return_value="/usr/bin/claude"), patch(
        "filings_analyst.providers.subprocess.run", return_value=fake
    ):
        p = providers.LLMProvider(provider="claude_cli")
        assert p.generate("hi") is None


def test_claude_cli_timeout_returns_none():
    with patch("filings_analyst.providers.shutil.which", return_value="/usr/bin/claude"), patch(
        "filings_analyst.providers.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=10),
    ):
        p = providers.LLMProvider(provider="claude_cli")
        assert p.generate("hi") is None


def test_anthropic_api_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    fake_resp = MagicMock(
        ok=True,
        status_code=200,
    )
    fake_resp.json.return_value = {"content": [{"text": "anthropic reply"}]}
    with patch("filings_analyst.providers.requests.post", return_value=fake_resp) as mock_post:
        p = providers.LLMProvider(provider="anthropic_api")
        out = p.generate("question", system="sys")
        assert out == "anthropic reply"
        body = mock_post.call_args.kwargs["json"]
        assert body["system"] == "sys"
        assert body["messages"][0]["content"] == "question"
        assert mock_post.call_args.kwargs["headers"]["x-api-key"] == "sk-fake"


def test_anthropic_api_http_error_returns_none(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    fake_resp = MagicMock(ok=False, status_code=500, text="server boom")
    with patch("filings_analyst.providers.requests.post", return_value=fake_resp):
        p = providers.LLMProvider(provider="anthropic_api")
        assert p.generate("q") is None


def test_openai_api_success(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    fake_resp = MagicMock(ok=True, status_code=200)
    fake_resp.json.return_value = {"choices": [{"message": {"content": "openai reply"}}]}
    with patch("filings_analyst.providers.requests.post", return_value=fake_resp) as mock_post:
        p = providers.LLMProvider(provider="openai_api")
        out = p.generate("q", system="sys")
        assert out == "openai reply"
        body = mock_post.call_args.kwargs["json"]
        # System message goes in messages[0].
        assert body["messages"][0]["role"] == "system"
        assert body["temperature"] == 0
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-fake"
