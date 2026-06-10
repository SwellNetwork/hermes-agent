"""Unit tests for agent/hindsight_session_hook.py — SQLite-native memory hook.

Covers the file-read based implementation that replaced the external Hindsight
API integration. No network calls are made; all tests use temp files on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_memory_file(path: Path, entries: List[str]) -> None:
    """Write a Hermes-format memory file with entries delimited by \\n§\\n."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n§\n".join(entries), encoding="utf-8")


# ---------------------------------------------------------------------------
# _read_memory_file()
# ---------------------------------------------------------------------------

def test_read_memory_file_missing_returns_empty(tmp_path):
    from agent.hindsight_session_hook import _read_memory_file

    result = _read_memory_file(tmp_path / "MISSING.md")
    assert result == []


def test_read_memory_file_empty_file_returns_empty(tmp_path):
    from agent.hindsight_session_hook import _read_memory_file

    p = tmp_path / "MEMORY.md"
    p.write_text("", encoding="utf-8")
    assert _read_memory_file(p) == []


def test_read_memory_file_whitespace_only_returns_empty(tmp_path):
    from agent.hindsight_session_hook import _read_memory_file

    p = tmp_path / "MEMORY.md"
    p.write_text("   \n  \n", encoding="utf-8")
    assert _read_memory_file(p) == []


def test_read_memory_file_single_entry(tmp_path):
    from agent.hindsight_session_hook import _read_memory_file

    p = tmp_path / "MEMORY.md"
    _write_memory_file(p, ["User prefers concise responses."])
    result = _read_memory_file(p)
    assert result == ["User prefers concise responses."]


def test_read_memory_file_multiple_entries(tmp_path):
    from agent.hindsight_session_hook import _read_memory_file

    entries = ["First memory.", "Second memory.", "Third memory."]
    p = tmp_path / "MEMORY.md"
    _write_memory_file(p, entries)
    result = _read_memory_file(p)
    assert result == entries


def test_read_memory_file_strips_whitespace_from_entries(tmp_path):
    from agent.hindsight_session_hook import _read_memory_file

    p = tmp_path / "MEMORY.md"
    p.write_text("  entry one  \n§\n  entry two  ", encoding="utf-8")
    result = _read_memory_file(p)
    assert result == ["entry one", "entry two"]


# ---------------------------------------------------------------------------
# _format_entries()
# ---------------------------------------------------------------------------

def test_format_entries_empty():
    from agent.hindsight_session_hook import _format_entries

    assert _format_entries([]) == ""


def test_format_entries_single():
    from agent.hindsight_session_hook import _format_entries

    assert _format_entries(["one"]) == "- one"


def test_format_entries_multiple():
    from agent.hindsight_session_hook import _format_entries

    result = _format_entries(["first", "second", "third"])
    assert result == "- first\n- second\n- third"


# ---------------------------------------------------------------------------
# recall_memories_for_prompt()
# ---------------------------------------------------------------------------

def test_recall_returns_empty_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.hindsight_session_hook import recall_memories_for_prompt

    result = recall_memories_for_prompt()
    assert result == ""


def test_recall_returns_platform_context_from_memory_md(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories_dir = tmp_path / "memories"
    _write_memory_file(memories_dir / "MEMORY.md", ["Faro is a trading platform."])

    from agent.hindsight_session_hook import recall_memories_for_prompt

    result = recall_memories_for_prompt()
    assert "## Platform Context" in result
    assert "Faro is a trading platform." in result
    assert "- Faro is a trading platform." in result


def test_recall_returns_user_context_from_user_md(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories_dir = tmp_path / "memories"
    _write_memory_file(memories_dir / "USER.md", ["User prefers bullet lists."])

    from agent.hindsight_session_hook import recall_memories_for_prompt

    result = recall_memories_for_prompt()
    assert "## User Context" in result
    assert "User prefers bullet lists." in result
    assert "Platform Context" not in result


def test_recall_returns_both_sections_when_both_files_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories_dir = tmp_path / "memories"
    _write_memory_file(memories_dir / "MEMORY.md", ["Domain fact."])
    _write_memory_file(memories_dir / "USER.md", ["User preference."])

    from agent.hindsight_session_hook import recall_memories_for_prompt

    result = recall_memories_for_prompt()
    assert "## Platform Context" in result
    assert "Domain fact." in result
    assert "## User Context" in result
    assert "User preference." in result


def test_recall_multiple_memory_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories_dir = tmp_path / "memories"
    _write_memory_file(memories_dir / "MEMORY.md", ["Entry A.", "Entry B.", "Entry C."])

    from agent.hindsight_session_hook import recall_memories_for_prompt

    result = recall_memories_for_prompt()
    assert "- Entry A." in result
    assert "- Entry B." in result
    assert "- Entry C." in result


def test_recall_accepts_user_message_and_user_id_without_error(tmp_path, monkeypatch):
    """user_message and user_id params are accepted for API compatibility but ignored."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.hindsight_session_hook import recall_memories_for_prompt

    # Should not raise regardless of what is passed
    result = recall_memories_for_prompt(user_message="hello", user_id="u123")
    assert result == ""


def test_recall_empty_memory_file_skips_platform_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories_dir = tmp_path / "memories"
    (memories_dir).mkdir(parents=True, exist_ok=True)
    (memories_dir / "MEMORY.md").write_text("", encoding="utf-8")
    _write_memory_file(memories_dir / "USER.md", ["User preference."])

    from agent.hindsight_session_hook import recall_memories_for_prompt

    result = recall_memories_for_prompt()
    assert "Platform Context" not in result
    assert "## User Context" in result


# ---------------------------------------------------------------------------
# emit_session_end() — no-op shim
# ---------------------------------------------------------------------------

def test_emit_session_end_is_noop_no_raise():
    """emit_session_end must not raise regardless of args."""
    from agent.hindsight_session_hook import emit_session_end

    emit_session_end(session_id="s1", user_id="u1", transcript=[])


def test_emit_session_end_accepts_none_args():
    from agent.hindsight_session_hook import emit_session_end

    emit_session_end(session_id="s2", user_id=None, transcript=None)


def test_emit_session_end_accepts_no_optional_args():
    from agent.hindsight_session_hook import emit_session_end

    emit_session_end(session_id="s3")


def test_emit_session_end_accepts_transcript_list():
    from agent.hindsight_session_hook import emit_session_end

    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    emit_session_end(session_id="s4", transcript=transcript)


# ---------------------------------------------------------------------------
# build_hindsight_prompt_fragment()
# ---------------------------------------------------------------------------

def test_build_prompt_fragment_returns_empty_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.hindsight_session_hook import build_hindsight_prompt_fragment

    result = build_hindsight_prompt_fragment(user_message="hello", user_id="u1")
    assert result == ""


def test_build_prompt_fragment_returns_content_when_files_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories_dir = tmp_path / "memories"
    _write_memory_file(memories_dir / "MEMORY.md", ["Platform info."])

    from agent.hindsight_session_hook import build_hindsight_prompt_fragment

    result = build_hindsight_prompt_fragment(user_message="hello", user_id="u1")
    assert "## Platform Context" in result
    assert "Platform info." in result


def test_build_prompt_fragment_delegates_to_recall(tmp_path, monkeypatch):
    """build_hindsight_prompt_fragment must forward to recall_memories_for_prompt."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories_dir = tmp_path / "memories"
    _write_memory_file(memories_dir / "MEMORY.md", ["Faro fact."])
    _write_memory_file(memories_dir / "USER.md", ["User pref."])

    from agent.hindsight_session_hook import (
        build_hindsight_prompt_fragment,
        recall_memories_for_prompt,
    )

    fragment = build_hindsight_prompt_fragment()
    direct = recall_memories_for_prompt()
    assert fragment == direct


# ---------------------------------------------------------------------------
# Integration: system_prompt.py injection (lightweight import checks)
# ---------------------------------------------------------------------------

def test_system_prompt_imports_hook_without_error():
    """system_prompt.py can import build_hindsight_prompt_fragment without error."""
    import agent.system_prompt  # noqa: F401 — import side-effect is the test
    from agent.hindsight_session_hook import build_hindsight_prompt_fragment

    assert callable(build_hindsight_prompt_fragment)


def test_system_prompt_hook_injection_code_path_exists():
    """Verify the hook injection code path exists in system_prompt.py."""
    import agent.system_prompt as sp
    import inspect

    source = inspect.getsource(sp.build_system_prompt_parts)
    assert "build_hindsight_prompt_fragment" in source
    assert "_user_id" in source


def test_hook_returns_empty_when_hermes_home_has_no_memories(tmp_path, monkeypatch):
    """When HERMES_HOME/memories/ is empty, hook returns ''."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "memories").mkdir()

    from agent.hindsight_session_hook import build_hindsight_prompt_fragment

    result = build_hindsight_prompt_fragment()
    assert result == ""
