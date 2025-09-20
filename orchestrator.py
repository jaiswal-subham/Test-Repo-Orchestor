# app/orchestrator_service.py
import logging
from typing import Dict, Any
from pydantic import BaseModel
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from init_pdf import init_pdf_node
from beneficiary_agent import beneficiary_agent_node
from provider_agent import provider_agent_node
from llm_utils import call_llm_json

logger = logging.getLogger("orchestrator_service")

# Pydantic schema for orchestrator structured output (keep name compatible with prior code)
class OrchestratorOutput(BaseModel):
    route: str

# Orchestrator decision instructions
ORCH_SYS = """
You are an orchestrator. 
Decide the next route for a multi-agent system.
Valid routes are:
  - "provider"    -> when user asks to find/compare/book providers (doctors, clinics)
  - "beneficiary" -> when user asks about beneficiary details in a PDF
  - "finalize"    -> when enough info is gathered to give a final answer or when user asks something unrelated
Return strict JSON, e.g.: {"route": "provider"}
""".strip()

class AgentState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    route: Literal["provider", "beneficiary", "finalize"]
    pdf_text: str
    provider_candidates: Optional[List[Dict[str, Any]]]
    provider_last_updated: Optional[int]
    beneficiary_answer: Optional[Dict[str, Any]]
    beneficiary_last_updated: Optional[int]
    final_answer: Optional[str]



# Node function: decide route
def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_query = msg.content
            break
    payload = call_llm_json(ORCH_SYS, user_query, schema=OrchestratorOutput)
    route = payload.get("route", "finalize")
    logger.info("Orchestrator decided route -> %s", route)
    return {"route": route}

# Finalize node: combine last agent outputs into final assistant reply
# put this helper near the top of your orchestrator module (or a shared utils file)
def _extract_msg_fields(msg):
    """
    Returns a tuple (role, content, metadata_dict)
    Works for:
      - plain dict messages (with keys like "role", "content", "agent", "response_key")
      - LangChain/Graph message objects (HumanMessage/AIMessage) which have .type, .content, .metadata
    Normalizes role to one of: "user"|"assistant"|"system"|"human" (we treat "human" same as "user").
    """
    # dict-like
    if isinstance(msg, dict):
        role = msg.get("role")
        content = msg.get("content")
        # metadata could be stored directly on dict
        metadata = {k: v for k, v in msg.items() if k not in ("role", "content")}
        return role, content, metadata

    # object-like (LangChain message)
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    content = getattr(msg, "content", None)
    # langchain message stores extra fields in .metadata often
    metadata = {}
    meta_obj = getattr(msg, "metadata", None)
    if isinstance(meta_obj, dict):
        metadata.update(meta_obj)
    # also check attributes directly on the object for agent/response_key (defensive)
    if hasattr(msg, "agent"):
        metadata["agent"] = getattr(msg, "agent")
    if hasattr(msg, "response_key"):
        metadata["response_key"] = getattr(msg, "response_key")

    # normalize role names (some libs use "human", some "user", some "ai")
    if role:
        r = role.lower()
        if "human" in r:
            role = "user"
        elif "assistant" in r or "ai" in r:
            role = "assistant"
    else:
        # fallback to class name
        clsname = msg.__class__.__name__.lower()
        if "human" in clsname:
            role = "user"
        elif "ai" in clsname or "assistant" in clsname:
            role = "assistant"

    return role, content, metadata

def _get_message_content(msg) -> str:
    """Extract content safely from dict or LangChain message objects."""
    if hasattr(msg, "content"):
        return msg.content
    if isinstance(msg, dict):
        return msg.get("content", "")
    return str(msg)

def _get_message_agent(msg) -> str | None:
    """Extract agent tag if available."""
    if isinstance(msg, dict):
        return msg.get("agent")
    if hasattr(msg, "metadata") and isinstance(msg.metadata, dict):
        return msg.metadata.get("agent")
    if hasattr(msg, "agent"):
        return getattr(msg, "agent")
    return None

def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages", []) or []

    # find last assistant message
    last_assistant = None
    for m in reversed(msgs):
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
        if role == "assistant":
            last_assistant = m
            break

    if last_assistant:
        reply = _get_message_content(last_assistant)
    else:
        reply = "I couldn’t extract relevant details. Please try refining your query."

    # wrap it in a finalizer message
    new_msg = {"role": "assistant", "content": reply, "agent": "finalizer"}
    msgs = msgs + [new_msg]

    return {"messages": msgs, "final_answer": reply}

def build_graph():
    graph_builder = StateGraph(AgentState)  # we accept generic dict states
    graph_builder.add_node("init_pdf", init_pdf_node)
    graph_builder.add_node("orchestrator", orchestrator_node)
    graph_builder.add_node("provider_agent", provider_agent_node)
    graph_builder.add_node("beneficiary_agent", beneficiary_agent_node)
    graph_builder.add_node("finalize", finalize_node)

    graph_builder.add_edge(START, "init_pdf")
    graph_builder.add_edge("init_pdf", "orchestrator")
    graph_builder.add_conditional_edges(
        "orchestrator",
        lambda state: state["route"],
        {"provider": "provider_agent", "beneficiary": "beneficiary_agent", "finalize": "finalize"},
    )
    graph_builder.add_edge("provider_agent", "finalize")
    graph_builder.add_edge("beneficiary_agent", "finalize")
    graph_builder.add_edge("init_pdf", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph
    

_graph = build_graph()

# snippet inside app/orchestrator_service.py (replace the run_orchestrator function)
def run_orchestrator(messages: list, doc_text: str | None = None) -> Dict[str, Any]:
    """
    messages: list of {"role": "user"|"human", "content": "..."}
    doc_text: optional PDF text to include in the state so beneficiary agent can use it
    returns the final state (dict) after graph.invoke
    """
    initial_state = {"messages": messages}
    if doc_text:
        # put PDF text into state as `pdf_text` so beneficiary_agent_node can read it
        initial_state["pdf_text"] = doc_text
    result = _graph.invoke(initial_state, config={"configurable": {"thread_id": "demo-thread"}})
    return result

