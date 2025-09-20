# app/llm_utils.py
import os
import logging
from typing import Any, Dict, Type, Optional
from openai import OpenAI  # OpenAI API client
from langchain_openai import ChatOpenAI  # LLM wrapper

logger = logging.getLogger("llm_utils")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1) 


# This function mirrors your previous call_llm_json but includes safe fallback
def call_llm_json(system: str, user: str, schema: Optional[Type] = None) -> Dict[str, Any]:
    """
    If schema is provided and llm supports with_structured_output, try to validate
    and return dict. Otherwise return a simple text summary in {'summary': ...}.
    If no LLM available, return a heuristic fallback.
    """
   
    try:
        if schema:
            structured_llm = llm.with_structured_output(schema)
            resp = structured_llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            # resp may be a pydantic model
            if hasattr(resp, "dict"):
                return resp.dict()
            # otherwise try to coerce
            return dict(resp)
        else:
            resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            # many wrappers provide .content
            if hasattr(resp, "content"):
                return {"summary": resp.content}
            return {"summary": str(resp)}
    except Exception as e:
        logger.exception("LLM call failed, returning fallback. Error: %s", e)
        return {"summary": "[LLM error] " + str(e)}
