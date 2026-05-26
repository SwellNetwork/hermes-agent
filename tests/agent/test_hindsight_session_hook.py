"""Unit tests for agent/hindsight_session_hook.py — Faro memory integration."""

import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module reload helper — ensures env vars are re-read in each test.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_env_and_reload():
    """Reset env vars and reload the module before each test."""
    for key in ("HINDSIGHT_URL", "FARO_MEMORY_SERVICE_URL"):
        os.environ.pop(key, None)

    # Clear the module cache so it reloads fresh.
    import agent.hindsight_session_hook as mod
    mod._client = None
    mod._ENABLED = False
    mod._HINDSIGHT_URL = ""
    mod._FARO_MEMORY_SERVICE_URL = ""
    mod._resolve_env()
    yield


# ---------------------------------------------------------------------------
# recall_memories_for_prompt()
# ---------------------------------------------------------------------------

def test_recall_disabled_when_no_hindsight_url():
    """recall_memories_for_prompt returns empty string when HINDSIGHT_URL is unset."""
    from agent.hindsight_session_hook import recall_memories_for_prompt

    result = recall_memories_for_prompt(user_message="hello", user_id="u1")
    assert result == ""


def test_recall_disabled_when_client_import_fails():
    """recall_memories_for_prompt returns empty string when hindsight_client is missing."""
    os.environ["HINDSIGHT_URL"] = "http://hindsight:8080"

    from agent.hindsight_session_hook import recall_memories_for_prompt, _resolve_env

    _resolve_env()

    with patch(
        "agent.hindsight_session_hook._get_hindsight_client", return_value=None
    ):
        result = recall_memories_for_prompt(user_message="hello", user_id="u1")
        assert result == ""


def test_recall_returns_formatted_memories():
    """recall_memories_for_prompt formats memories from both banks correctly."""
    os.environ["HINDSIGHT_URL"] = "http://hindsight:8080"

    from agent.hindsight_session_hook import (
        recall_memories_for_prompt,
        _resolve_env,
        _get_hindsight_client,
    )

    _resolve_env()

    # Create a mock client whose recall() returns pre-built Memory objects.
    fake_mem1 = MagicMock()
    fake_mem1.content = "Faro is a trading platform."
    fake_mem2 = MagicMock()
    fake_mem2.content = "User prefers concise responses."

    mock_client = MagicMock()

    async def _fake_recall(query, bank_id, top_k):
        if bank_id == "faro":
            return [fake_mem1]
        elif bank_id.startswith("user-"):
            return [fake_mem2]
        return []

    mock_client.recall = _fake_recall

    with patch.object(
        __import__("agent.hindsight_session_hook", fromlist=["_get_hindsight_client"]),
        "_get_hindsight_client",
        return_value=mock_client,
    ):
        result = recall_memories_for_prompt(
            user_message="What is Faro?", user_id="u1"
        )

    assert "Platform Context" in result
    assert "Faro is a trading platform" in result
    assert "User Context" in result
    assert "User prefers concise responses" in result


def test_recall_handles_partial_failure():
    """When only one bank succeeds, the other is omitted gracefully."""
    os.environ["HINDSIGHT_URL"] = "http://hindsight:8080"

    from agent.hindsight_session_hook import (
        recall_memories_for_prompt,
        _resolve_env,
    )

    _resolve_env()

    fake_mem = MagicMock()
    fake_mem.content = "Faro platform info."

    mock_client = MagicMock()

    async def _fake_recall(query, bank_id, top_k):
        if bank_id == "faro":
            return [fake_mem]
        raise RuntimeError("user bank unavailable")

    mock_client.recall = _fake_recall

    with patch.object(
        __import__("agent.hindsight_session_hook", fromlist=["_get_hindsight_client"]),
        "_get_hindsight_client",
        return_value=mock_client,
    ):
        result = recall_memories_for_prompt(
            user_message="What is Faro?", user_id="u1"
        )

    # Should have domain context but no user context (the second recall failed).
    assert "Platform Context" in result
    assert "User Context" not in result


def test_recall_handles_both_banks_empty():
    """When both banks return empty lists, the result is empty."""
    os.environ["HINDSIGHT_URL"] = "http://hindsight:8080"

    from agent.hindsight_session_hook import (
        recall_memories_for_prompt,
        _resolve_env,
    )

    _resolve_env()

    mock_client = MagicMock()

    async def _fake_recall(query, bank_id, top_k):
        return []

    mock_client.recall = _fake_recall

    with patch.object(
        __import__("agent.hindsight_session_hook", fromlist=["_get_hindsight_client"]),
        "_get_hindsight_client",
        return_value=mock_client,
    ):
        result = recall_memories_for_prompt(
            user_message="What is Faro?", user_id="u1"
        )

    assert result == ""


def test_recall_empty_user_message_and_id():
    """recall works with empty/None user_message and user_id."""
    os.environ["HINDSIGHT_URL"] = "http://hindsight:8080"

    from agent.hindsight_session_hook import (
        recall_memories_for_prompt,
        _resolve_env,
    )

    _resolve_env()

    fake_mem = MagicMock()
    fake_mem.content = "general info"

    mock_client = MagicMock()

    async def _fake_recall(query, bank_id, top_k):
        return [fake_mem]

    mock_client.recall = _fake_recall

    with patch.object(
        __import__("agent.hindsight_session_hook", fromlist=["_get_hindsight_client"]),
        "_get_hindsight_client",
        return_value=mock_client,
    ):
        result = recall_memories_for_prompt(user_message=None, user_id=None)

    # Even with empty query, both banks should be queried.
    assert "general info" in result


# ---------------------------------------------------------------------------
# _format_memories()
# ---------------------------------------------------------------------------

def test_format_memories_empty():
    from agent.hindsight_session_hook import _format_memories

    assert _format_memories([]) == "_No relevant memories found._"


def test_format_memories_single():
    from agent.hindsight_session_hook import _format_memories

    m = MagicMock()
    m.content = "single memory"
    assert _format_memories([m]) == "- single memory"


def test_format_memories_multiple():
    from agent.hindsight_session_hook import _format_memories

    m1 = MagicMock()
    m1.content = "first"
    m2 = MagicMock()
    m2.content = "second"
    assert _format_memories([m1, m2]) == "- first\n- second"


def test_format_memories_fallback_to_str():
    """When content attr is missing, fall back to repr."""
    from agent.hindsight_session_hook import _format_memories

    result = _format_memories(["raw string"])
    assert "- raw string" in result


# ---------------------------------------------------------------------------
# emit_session_end()
# ---------------------------------------------------------------------------

def test_emit_session_end_skips_when_url_unset():
    """emit_session_end does nothing when FARO_MEMORY_SERVICE_URL is empty."""
    from agent.hindsight_session_hook import emit_session_end

    # Should not raise
    emit_session_end(session_id="s1", user_id="u1", transcript=[])


def test_emit_session_end_posts_to_endpoint():
    """emit_session_end POSTs the transcript to faro-memory-service."""
    os.environ["FARO_MEMORY_SERVICE_URL"] = "http://faro-memory:8080"

    from agent.hindsight_session_hook import emit_session_end, _resolve_env

    _resolve_env()

    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        emit_session_end(
            session_id="sess-abc",
            user_id="user-123",
            transcript=transcript,
        )

        # Wait for the daemon thread to complete.
        import time
        time.sleep(0.1)

        mock_urlopen.assert_called_once()

        # Verify the request URL.
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://faro-memory:8080/extract"
        assert req.method == "POST"
        assert req.get_header("Content-type") == "application/json"


def test_emit_session_end_survives_connection_error():
    """emit_session_end does not raise when the HTTP call fails."""
    os.environ["FARO_MEMORY_SERVICE_URL"] = "http://faro-memory:8080"

    from agent.hindsight_session_hook import emit_session_end, _resolve_env

    _resolve_env()

    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        # Should not raise — fire-and-forget
        emit_session_end(
            session_id="sess-abc",
            user_id="user-123",
            transcript=[],
        )

        import time
        time.sleep(0.1)


def test_emit_session_end_with_none_transcript():
    """emit_session_end handles None transcript gracefully."""
    os.environ["FARO_MEMORY_SERVICE_URL"] = "http://faro-memory:8080"

    from agent.hindsight_session_hook import emit_session_end, _resolve_env

    _resolve_env()

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        emit_session_end(session_id="s1", user_id=None, transcript=None)

        import time
        time.sleep(0.1)

        mock_urlopen.assert_called_once()


# ---------------------------------------------------------------------------
# build_hindsight_prompt_fragment()
# ---------------------------------------------------------------------------

def test_build_prompt_fragment_disabled():
    """build_hindsight_prompt_fragment returns '' when HINDSIGHT_URL is unset."""
    from agent.hindsight_session_hook import build_hindsight_prompt_fragment

    result = build_hindsight_prompt_fragment(
        user_message="hello", user_id="u1"
    )
    assert result == ""


def test_build_prompt_fragment_returns_block():
    """build_hindsight_prompt_fragment forwards to recall_memories_for_prompt."""
    os.environ["HINDSIGHT_URL"] = "http://hindsight:8080"

    from agent.hindsight_session_hook import (
        build_hindsight_prompt_fragment,
        _resolve_env,
    )

    _resolve_env()

    with patch(
        "agent.hindsight_session_hook.recall_memories_for_prompt",
        return_value="## Platform Context\n- test",
    ):
        result = build_hindsight_prompt_fragment(
            user_message="hello", user_id="u1"
        )
        assert result == "## Platform Context\n- test"


# ---------------------------------------------------------------------------
# Integration: system_prompt.py injection (lightweight)
# ---------------------------------------------------------------------------

def test_system_prompt_imports_hindsight_hook():
    """system_prompt.py can import build_hindsight_prompt_fragment without error."""
    # Verify the import works — the system_prompt module loads without
    # crashing even when HINDSIGHT_URL is unset.
    import agent.system_prompt
    from agent.hindsight_session_hook import build_hindsight_prompt_fragment
    assert callable(build_hindsight_prompt_fragment)


def test_system_prompt_hindsight_block_not_injected_when_disabled():
    """When HINDSIGHT_URL is unset, build_hindsight_prompt_fragment returns ''."""
    from agent.hindsight_session_hook import build_hindsight_prompt_fragment

    result = build_hindsight_prompt_fragment(
        user_message="hello", user_id="u1"
    )
    assert result == ""


def test_system_prompt_hindsight_injection_code_path_exists():
    """Verify the injection code at line ~296 in system_prompt.py exists."""
    import agent.system_prompt as sp
    import inspect

    source = inspect.getsource(sp.build_system_prompt_parts)
    assert "build_hindsight_prompt_fragment" in source
    assert "_user_id" in source
