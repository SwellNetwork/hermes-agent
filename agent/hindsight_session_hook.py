"""Hindsight session hook — Faro memory integration.

Provides session-start recall and session-end retention hooks that
integrate Hindsight long-term memory into every Hermes session.

At session start:
  1. Recall from ``faro`` bank (domain/platform knowledge)
  2. Recall from ``user-{user_id}`` bank (per-user preferences)
  3. Format memories and inject into the system prompt volatile tier

At session end:
  1. POST the full transcript to ``FARO_MEMORY_SERVICE_URL/extract``
     for async extraction and retention into Hindsight banks.

Configuration via environment variables:
  ``HINDSIGHT_URL``          — Hindsight API endpoint (ClusterIP in K8s)
  ``FARO_MEMORY_SERVICE_URL`` — Memory extraction service endpoint

Both hooks are fire-and-forget: failures are logged but never block
the agent loop or the user's response.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-process lazy client singleton — avoids re-resolving on every turn
# ---------------------------------------------------------------------------

_client: Any = None
_client_lock = threading.Lock()

_FARO_MEMORY_SERVICE_URL = ""
_HINDSIGHT_URL = ""

# Whether the session-start recall hook is enabled.
# Disabled when HINDSIGHT_URL is unset (graceful no-op).
_ENABLED = False


def _resolve_env() -> None:
    """One-time resolution of HINDSIGHT_URL and FARO_MEMORY_SERVICE_URL."""
    global _HINDSIGHT_URL, _FARO_MEMORY_SERVICE_URL, _ENABLED

    _HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL", "").strip().rstrip("/")
    _FARO_MEMORY_SERVICE_URL = (
        os.environ.get("FARO_MEMORY_SERVICE_URL", "").strip().rstrip("/")
    )
    _ENABLED = bool(_HINDSIGHT_URL)


_resolve_env()


def _get_hindsight_client():
    """Return a lazily-created Hindsight client, or None if unavailable."""
    global _client, _ENABLED
    if not _ENABLED:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from hindsight_client import Hindsight
            _client = Hindsight(url=_HINDSIGHT_URL)
            logger.info(
                "Hindsight session hook initialised (url=%s)", _HINDSIGHT_URL
            )
        except ImportError:
            logger.warning(
                "hindsight-client not installed — session hook disabled"
            )
            _ENABLED = False
            return None
        except Exception as exc:
            logger.warning(
                "Failed to initialise Hindsight client: %s — hook disabled", exc
            )
            _ENABLED = False
            return None
        return _client


# ---------------------------------------------------------------------------
# Session-start recall
# ---------------------------------------------------------------------------

def _format_memory(mem: Any) -> str:
    """Format a single Hindsight memory entry into a one-line context string."""
    try:
        content = getattr(mem, "content", None) or str(mem)
        return f"- {content}"
    except Exception:
        return f"- {mem!r}"


def _format_memories(memories: List[Any]) -> str:
    """Format a list of Hindsight memory results into a prompt block."""
    if not memories:
        return "_No relevant memories found._"
    return "\n".join(_format_memory(m) for m in memories)


def recall_memories_for_prompt(
    user_message: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Recall memories from both Hindsight banks and format them for the prompt.

    The two recalls run in parallel via ``asyncio.gather`` using the
    existing Hindsight async event loop.  If either recall fails, the
    other continues independently.

    Returns:
        A formatted string to inject into the system prompt volatile tier,
        or an empty string if HINDSIGHT_URL is unset or both recalls fail.
    """
    if not _ENABLED:
        return ""

    client = _get_hindsight_client()
    if client is None:
        return ""

    query = (user_message or "").strip()
    uid = (user_id or "").strip()

    # Import asyncio lazily so the module loads without an event loop.
    import asyncio

    domain_mems: List[Any] = []
    user_mems: List[Any] = []

    async def _recall():
        nonlocal domain_mems, user_mems
        results = await asyncio.gather(
            client.recall(query=query, bank_id="faro", top_k=10),
            client.recall(
                query=query, bank_id=f"user-{uid}", top_k=10
            ),
            return_exceptions=True,
        )
        # Unpack results, treating exceptions as empty lists.
        _domain, _user = results
        if isinstance(_domain, list):
            domain_mems = _domain
        elif isinstance(_domain, Exception):
            logger.warning("Hindsight recall (faro bank) failed: %s", _domain)
        if isinstance(_user, list):
            user_mems = _user
        elif isinstance(_user, Exception):
            logger.warning("Hindsight recall (user bank) failed: %s", _user)

    try:
        # Use the shared Hindsight event loop if available; otherwise create one.
        from plugins.memory.hindsight import _get_loop as _hindsight_loop
        loop = _hindsight_loop()
    except ImportError:
        loop = None

    if loop is not None and loop.is_running():
        from agent.async_utils import safe_schedule_threadsafe
        fut = safe_schedule_threadsafe(_recall(), loop)
        if fut is not None:
            try:
                fut.result(timeout=30)
            except Exception as exc:
                logger.warning("Hindsight recall timed out or failed: %s", exc)
    else:
        # Fallback: run in a fresh event loop on the current thread.
        try:
            asyncio.run(_recall())
        except RuntimeError:
            # Already inside an event loop — run synchronously in a thread.
            def _run_recall():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_recall())
                finally:
                    loop.close()

            t = threading.Thread(target=_run_recall, daemon=True)
            t.start()
            t.join(timeout=30)
            if t.is_alive():
                logger.warning("Hindsight recall timed out (30s)")

    parts: List[str] = []
    if domain_mems:
        parts.append(
            "## Platform Context (Faro)\n"
            + _format_memories(domain_mems)
        )
    if user_mems:
        parts.append(
            "## User Context\n"
            + _format_memories(user_mems)
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Session-end retention
# ---------------------------------------------------------------------------

def emit_session_end(
    session_id: str,
    user_id: Optional[str] = None,
    transcript: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Emit the session transcript to faro-memory-service for async extraction.

    This is a fire-and-forget HTTP POST — it spawns a daemon thread so
    the caller never blocks.  Failures are logged at warning level but
    never raised to the caller.

    Args:
        session_id: The Hermes session identifier.
        user_id: The platform user identifier (if known).
        transcript: The full conversation transcript as a list of message dicts.
    """
    url = _FARO_MEMORY_SERVICE_URL
    if not url:
        logger.debug(
            "FARO_MEMORY_SERVICE_URL not set — session-end retention skipped"
        )
        return

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id or "",
        "transcript": transcript or [],
    }

    def _post():
        try:
            import urllib.request
            import json as _json

            data = _json.dumps(payload, default=str).encode("utf-8")
            req = urllib.request.Request(
                f"{url}/extract",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                status = resp.status
            logger.debug(
                "faro-memory-service /extract responded HTTP %d "
                "(session=%s)",
                status,
                session_id,
            )
        except Exception as exc:
            logger.warning(
                "faro-memory-service /extract failed (session=%s): %s",
                session_id,
                exc,
            )

    t = threading.Thread(target=_post, daemon=True, name="faro-memory-extract")
    t.start()


# ---------------------------------------------------------------------------
# Convenience: build the full prompt fragment for system-prompt assembly
# ---------------------------------------------------------------------------

def build_hindsight_prompt_fragment(
    user_message: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Build the Hindsight memory injection block for the system prompt.

    Returns a non-empty string only when HINDSIGHT_URL is set and at
    least one bank returns memories.

    This function is designed to be called from the volatile tier of
    ``build_system_prompt_parts`` in ``agent/system_prompt.py``.
    """
    return recall_memories_for_prompt(
        user_message=user_message,
        user_id=user_id,
    )
