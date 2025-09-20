# app/utility.py
from typing import Dict, Any, Optional, List


def _extract_content(msg) -> str:
    """Return the content of a message-like object/dict as a string."""
    if msg is None:
        return ""
    if hasattr(msg, "content"):
        return msg.content or ""
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return str(msg)


def get_last_human_message(state: Dict[str, Any]) -> Optional[str]:
    """
    Get the most recent human/user message from state["messages"].
    Accepts message items that are dicts (with 'role' or 'type' keys) or
    objects that expose .type/.role and .content attributes.
    Returns None if no human message found.
    """
    msgs: List = state.get("messages", []) or []
    for msg in reversed(msgs):
        role = None
        if hasattr(msg, "type"):
            role = getattr(msg, "type", None)
        elif hasattr(msg, "role"):
            role = getattr(msg, "role", None)
        elif isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
        if isinstance(role, str) and role.lower() in ("user", "human", "humanmessage"):
            return _extract_content(msg)
        if isinstance(role, str) and role.lower() == "user":
            # accept plain 'user' as well
            return _extract_content(msg)
    return None
