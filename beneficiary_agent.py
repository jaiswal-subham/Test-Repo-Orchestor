# app/agents/beneficiary_agent.py
import time
import logging
from typing import Dict, Any, Optional

from llm_utils import call_llm_json
from utility import get_last_human_message

logger = logging.getLogger("beneficiary_agent")


def _extract_text_from_llm_response(resp: Any) -> str:
    """
    Normalize a variety of LLM responses into a human-readable string content.
    If resp is a dict and contains 'summary', return that.
    If resp has 'content' attribute, return that.
    Otherwise fallback to str(resp).
    """
    if resp is None:
        return ""
    # dict with 'summary' key (our llm_utils uses this)
    if isinstance(resp, dict):
        if "summary" in resp and resp["summary"]:
            return str(resp["summary"])
        # if resp has keys but no summary, try common textual keys
        for k in ("content", "text", "answer"):
            if k in resp and resp[k]:
                return str(resp[k])
        # last resort: stringified dict
        return str(resp)
    # objects with content
    if hasattr(resp, "content"):
        return getattr(resp, "content") or ""
    return str(resp)


def beneficiary_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Answer based on PDF text + user question.

    Writes:
      - beneficiary_answer: the raw object returned by call_llm_json
      - beneficiary_last_updated: timestamp
      - final_answer: human-readable string extracted from LLM response
      - messages: assistant message with content == final_answer
    """
    pdf_text: str = state.get("pdf_text", "") or ""
    user_query: Optional[str] = get_last_human_message(state)

    if not pdf_text:
        answer = {"error": "No PDF data available"}
        final_answer = "No PDF data available to answer the question."
    else:
        answer = call_llm_json(
            system="You are Beneficiary Agent. Answer user queries using the PDF text.",
            user=f"PDF: {pdf_text}\n\nUser Question: {user_query}",
        )
        final_answer = _extract_text_from_llm_response(answer) or "No answer produced."

    return {
        "beneficiary_answer": answer,
        "beneficiary_last_updated": int(time.time()),
        "final_answer": final_answer,
        "messages": [
            {
                "role": "assistant",
                "content": final_answer,
                "agent": "beneficiary",
                "response_key": "beneficiary_answer",
            }
        ],
    }
