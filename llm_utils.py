# app/llm_utils.py
import os
import logging
from typing import Any, Dict, Type, Optional

# Try to import your LLM wrappers; keep graceful fallback if not available
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

logger = logging.getLogger("llm_utils")

client = None
if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        logger.exception("OpenAI client init failed")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = None
if ChatOpenAI is not None:
    try:
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1)
    except Exception:
        logger.exception("ChatOpenAI init failed")


def call_llm_json(system: str, user: str, schema: Optional[Type] = None) -> Dict[str, Any]:
    """
    If schema is provided and LLM supports with_structured_output, try to return dict.
    Otherwise return a dict with 'summary': textual content (or fallback).
    """
    try:
        if llm and schema:
            structured = llm.with_structured_output(schema)
            resp = structured.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            if hasattr(resp, "dict"):
                return resp.dict()
            # coerce to dict
            try:
                return dict(resp)
            except Exception:
                return {"summary": str(resp)}
        elif llm:
            resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            # prefer attribute 'content'
            if hasattr(resp, "content"):
                return {"summary": resp.content}
            # else return stringified
            return {"summary": str(resp)}
        else:
            # fallback when no LLM is configured
            # provide a short deterministic fallback summary so beneficiary_agent can set final_answer text
            truncated_user = (user or "")[:1000]
            return {"summary": f"[no-llm-fallback] {truncated_user}"}
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return {"summary": "[LLM error] " + str(e)}
