"""SQLite-native memory session hook — VM disk integration.

Replaces the external Hindsight dependency with direct reads from the VM
persistent disk files managed natively by Hermes (MEMORY.md, USER.md).

At session start:
  1. Read MEMORY.md from HERMES_HOME/memories/ and inject as ## Platform Context
  2. Read USER.md from HERMES_HOME/memories/ and inject as ## User Context

At session end:
  No-op — Hermes's native ``background_review`` handles memory extraction
  and persistence to ``state.db`` automatically.

No external service calls or environment variables required. Always active
when MEMORY.md or USER.md files exist on the VM disk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENTRY_DELIMITER = "\n§\n"


def _get_memory_dir() -> Path:
    """Return the Hermes memories directory.

    Resolves HERMES_HOME from the environment at call time so that
    per-test HERMES_HOME overrides are always respected.
    """
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if not hermes_home:
        hermes_home = str(Path.home() / ".hermes")
    return Path(hermes_home) / "memories"


def _read_memory_file(path: Path) -> List[str]:
    """Read a Hermes memory file and return a list of non-empty entries.

    Files are stored as entries delimited by ``\\n§\\n`` (the same format
    used by :class:`tools.memory_tool.MemoryStore`).  Missing or empty
    files return an empty list — never raise.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, IOError):
        return []

    if not raw.strip():
        return []

    entries = [e.strip() for e in raw.split(_ENTRY_DELIMITER)]
    return [e for e in entries if e]


def _format_entries(entries: List[str]) -> str:
    """Format memory entries as a bullet list."""
    return "\n".join(f"- {entry}" for entry in entries)


# ---------------------------------------------------------------------------
# Session-start recall
# ---------------------------------------------------------------------------

def recall_memories_for_prompt(
    user_message: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Read MEMORY.md and USER.md from disk and format for prompt injection.

    Reads ``HERMES_HOME/memories/MEMORY.md`` and
    ``HERMES_HOME/memories/USER.md`` and formats them as ``## Platform
    Context`` and ``## User Context`` sections respectively.

    Returns an empty string when both files are absent or empty — always
    safe to call regardless of whether the files exist.
    """
    mem_dir = _get_memory_dir()
    memory_entries = _read_memory_file(mem_dir / "MEMORY.md")
    user_entries = _read_memory_file(mem_dir / "USER.md")

    parts: List[str] = []
    if memory_entries:
        parts.append("## Platform Context\n" + _format_entries(memory_entries))
    if user_entries:
        parts.append("## User Context\n" + _format_entries(user_entries))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Session-end retention (no-op — delegated to Hermes background_review)
# ---------------------------------------------------------------------------

def emit_session_end(
    session_id: str,
    user_id: Optional[str] = None,
    transcript: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """No-op compatibility shim — memory persistence is handled natively.

    Hermes's built-in ``background_review`` extracts and persists memories
    to ``state.db`` at session end.  No external service call is needed.
    """
    logger.debug(
        "session-end hook: relying on native background_review for "
        "memory persistence (session=%s)",
        session_id,
    )


# ---------------------------------------------------------------------------
# Convenience: build the full prompt fragment for system-prompt assembly
# ---------------------------------------------------------------------------

def build_hindsight_prompt_fragment(
    user_message: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Build the memory injection block for the system prompt.

    Reads MEMORY.md and USER.md from the VM disk and formats them for
    injection into the volatile tier of the system prompt.  Returns an
    empty string when neither file has content.
    """
    return recall_memories_for_prompt(
        user_message=user_message,
        user_id=user_id,
    )
