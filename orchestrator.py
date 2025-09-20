# app/orchestrator_service.py
import logging
from typing import Dict, Any
from pydantic import BaseModel
from typing import TypedDict, List, Dict as TDict, Any as TAny, Optional, Literal
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from init_pdf import init_pdf_node
from beneficiary_agent import beneficiary_agent_node
from provider_agent import provider_agent_node
from llm_utils import call_llm_json
from utility import get_last_human_message

logger = logging.getLogger("orchestrator_service")

class OrchestratorOutput(BaseModel):
    route: str

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
    messages: Annotated[List[TDict[str, TAny]], add_messages]
    route: Literal["provider", "beneficiary", "finalize"]
    pdf_text: str
    provider_candidates: Optional[List[Dict[str, Any]]]
    provider_last_updated: Optional[int]
    beneficiary_answer: Optional[Dict[str, Any]]
    beneficiary_last_updated: Optional[int]
    final_answer: Optional[str]
    final_answer_last_updated: Optional[int]


def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = get_last_human_message(state)
    payload = call_llm_json(ORCH_SYS, user_query, schema=OrchestratorOutput)
    route = payload.get("route", "finalize")
    logger.info("Orchestrator decided route -> %s", route)
    return {"route": route}


def _get_message_content(msg) -> str:
    if hasattr(msg, "content"):
        return msg.content
    if isinstance(msg, dict):
        return msg.get("content", "")
    return str(msg)


def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalize: prefer explicit final_answer (string). If not present, derive from last assistant message.
    Returns { "messages": [ assistant message ], "final_answer": <string> } where final_answer is string.
    """
    msgs = state.get("messages", []) or []

    explicit_final = state.get("final_answer")
    if explicit_final is not None and isinstance(explicit_final, str):
        final_text = explicit_final
    else:
        # find last assistant message content
        last_assistant = None
        for m in reversed(msgs):
            role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
            if role == "assistant":
                last_assistant = m
                break
        if last_assistant:
            final_text = _get_message_content(last_assistant) or "I couldn't extract a final answer."
        else:
            final_text = "I couldn’t extract relevant details. Please try refining your query."

    final_msg = {"role": "assistant", "content": final_text, "agent": "finalizer"}
    return {"messages": [final_msg], "final_answer": final_text}


def build_graph():
    graph_builder = StateGraph(AgentState)
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


def run_orchestrator(messages: list, doc_text: str | None = None) -> Dict[str, Any]:
    initial_state = {"messages": messages}
    if doc_text:
        initial_state["pdf_text"] = doc_text
    result = _graph.invoke(initial_state, config={"configurable": {"thread_id": "demo-thread"}})
    return result
