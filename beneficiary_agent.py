# app/agents/beneficiary_agent.py
import time
import logging
from typing import Dict, Any

from llm_utils import call_llm_json
from utility import get_last_human_message

logger = logging.getLogger("beneficiary_agent")


def beneficiary_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Answer based on PDF text + user question. Returns partial state dict."""
   
    pdf_text = state.get("pdf_text", "")
    
    user_query = get_last_human_message(state)

    if not pdf_text:
        answer = {"error": "No PDF data available"}
    else:
        answer = call_llm_json(
            system="You are Beneficiary Agent. Answer user queries using the PDF text.",
            user=f"PDF: {pdf_text}\n\nUser Question: {user_query}",
        )

    return {
        "beneficiary_answer": answer,
        "beneficiary_last_updated": int(time.time()),
        "messages": [
            {
                "role": "assistant",
                "content": "Beneficiary Agent responded (see beneficiary_answer).",
                "agent": "beneficiary",
                "response_key": "beneficiary_answer",
            }
        ],
    }
