from typing import Dict, Any, Optional


def get_last_human_message(state: Dict[str, Any]) -> Optional[str]:
    """
    Utility to get the most recent human message from state["messages"].

    Args:
        state (Dict[str, Any]): A dict containing a "messages" list.

    Returns:
        Optional[str]: The content of the last human message, or None if not found.
    """
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", None) == "human":
            return getattr(msg, "content", None)
    return None