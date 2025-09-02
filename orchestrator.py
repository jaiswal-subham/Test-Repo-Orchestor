# app.py

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import json
import time

import logging
import requests
import gradio as gr  # UI framework for web apps
from dotenv import load_dotenv  # Load environment variables from .env
from typing import TypedDict, Annotated, Any, Dict, List, Literal, Optional, Type

from openai import OpenAI  # OpenAI API client
from pypdf import PdfReader  # PDF reading library
from pydantic import BaseModel  # Data validation & structured output
from langchain_openai import ChatOpenAI  # LLM wrapper
from langgraph.graph.message import add_messages  # For annotating messages in state
from langgraph.graph import StateGraph, START, END  # Graph-based workflow engine
from langgraph.checkpoint.memory import MemorySaver  # For storing graph checkpoints

from gradio import State  # Gradio state object

# -----------------------------
# Setup
# -----------------------------

# Load environment variables from a .env file
load_dotenv(override=True)

# Configure logging for the application
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("orchestrator")

# OpenAI API client & LLM setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1)  # Low temperature for deterministic output

logger.info("Setup complete. Using model=%s", DEFAULT_MODEL)

# -----------------------------
# Utilities
# -----------------------------

def load_pdf_text(pdf_path: str) -> str:
    """Read text from a PDF file and combine all pages into a single string."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# Preload a PDF file
PDF_PATH = "data/Axis-Max-STPP.pdf"
pdf_text = load_pdf_text(PDF_PATH)
print(pdf_text)

def call_llm_json(system: str, user: str, schema: Type = None) -> Dict[str, Any]:
    """
    Call the LLM to get structured output.
    - If a schema is provided, the LLM response is validated against it.
    - Otherwise, returns raw text.
    """
    if schema:
        structured_llm = llm.with_structured_output(schema)
        resp = structured_llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return resp.dict()  # Convert Pydantic model -> dict
    else:
        resp = llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return {"summary": resp.content if hasattr(resp, "content") else str(resp)}

# -----------------------------
# State and Node Definitions
# -----------------------------

# Define the structure of the AgentState dictionary
class AgentState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    route: Literal["offer", "beneficiary", "finalize"]
    pdf_text: str
    offer_summary: Optional[Dict[str, Any]]
    offer_last_updated: Optional[int]  # New
    beneficiary_answer: Optional[Dict[str, Any]]
    beneficiary_last_updated: Optional[int]  # New
    final_answer: Optional[str]

def init_pdf_node(state: AgentState) -> AgentState:
    """Initialize the state with preloaded PDF text."""
    return {"pdf_text": pdf_text}

# Orchestrator instructions for deciding the next node
ORCH_SYS = """
You are an orchestrator. 
Decide the next route for a multi-agent system.
Valid routes are:
  - "offer"       -> when user asks about live offers or prices (API)
  - "beneficiary" -> when user asks about beneficiary details in a PDF
  - "finalize"    -> when enough info is gathered to give a final answer or when the user asks something unrelated
Return strict JSON, e.g.: {"route": "offer"}
""".strip()

# Pydantic model for orchestrator structured output
class OrchestratorOutput(BaseModel):
    route: str

def orchestrator_node(state: AgentState) -> AgentState:
    """Decide next route in the workflow based on last user message."""
    user_query = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_query = msg.content
            break

    payload = call_llm_json(ORCH_SYS, user_query, schema=OrchestratorOutput)
    route = payload.get("route", "finalize")
    logger.info("Orchestrator decided route -> %s", route)
    return {"route": route}

# Offer agent node: fetch API data and summarize
def offer_agent_node(state: AgentState) -> AgentState:
    """Fetch offers from API and summarize for the user."""
    api_url = "https://64358969537112453fd91559.mockapi.io/offers"
    user_query = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_query = msg.content
            break
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        api_data = response.json()
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"API error: {e}"}]}

    summary = call_llm_json(
        system=(
            "You are Offer Agent. Based on the API response and user query, "
            "write a concise, user-friendly English explanation of the best matching offers."
        ),
        user=json.dumps({"query": user_query, "api_response": api_data}),
    )

    return {
        "offer_summary": summary,
        "offer_last_updated": int(time.time()),  # mark current timestamp
        "messages": [{"role": "assistant", "content": "Offer Agent processed API data."}],
    }


# Beneficiary agent: answer questions using PDF content
def beneficiary_agent_node(state: AgentState) -> AgentState:
    """Answer based on PDF text + user question."""
    pdf_text = state.get("pdf_text", "")
    last_message = state["messages"][-1]
    user_query = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_query = msg.content
            break
    print(user_query)
    if not pdf_text:
        answer = {"error": "No PDF data available"}
    else:
        answer = call_llm_json(
            system="You are Beneficiary Agent. Answer user queries using the PDF text.",
            user=f"PDF: {pdf_text}\n\nUser Question: {user_query}",
        )
    return {
        "beneficiary_answer": answer,
        "beneficiary_last_updated": int(time.time()),  # mark current timestamp
    }

# Finalize node: combine outputs into final assistant response
def finalize_node(state: AgentState) -> AgentState:
    last_agent_output = None
    offer_time = state.get("offer_last_updated", 0)
    beneficiary_time = state.get("beneficiary_last_updated", 0)

    if beneficiary_time >= offer_time and state.get("beneficiary_answer"):
        last_agent_output = ("beneficiary", state["beneficiary_answer"])
    elif offer_time > beneficiary_time and state.get("offer_summary"):
        last_agent_output = ("offer", state["offer_summary"])

    reply = ""
    if last_agent_output:
        agent_type, content = last_agent_output
        if agent_type == "offer":
            offer_text = content.get("summary") if isinstance(content, dict) else str(content)
            reply = f"📄 Offer Summary:\n{offer_text}"
        elif agent_type == "beneficiary":
            if isinstance(content, dict):
                beneficiary_text = content.get("summary") or content.get("error") or str(content)
            else:
                beneficiary_text = str(content)
            reply = f"👤 Beneficiary Info:\n{beneficiary_text}"
    else:
        reply = "I couldn’t extract relevant details. Please try refining your query."

    new_messages = state.get("messages", []) + [{"role": "assistant", "content": reply}]
    return {"messages": new_messages, "final_answer": reply}


# -----------------------------
# Graph Construction
# -----------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node("init_pdf", init_pdf_node)
graph_builder.add_node("orchestrator", orchestrator_node)
graph_builder.add_node("offer_agent", offer_agent_node)
graph_builder.add_node("beneficiary_agent", beneficiary_agent_node)
graph_builder.add_node("finalize", finalize_node)

# Define edges between nodes
graph_builder.add_edge(START, "init_pdf")
graph_builder.add_edge("init_pdf", "orchestrator")
graph_builder.add_conditional_edges(
    "orchestrator",
    lambda state: state["route"],
    {"offer": "offer_agent", "beneficiary": "beneficiary_agent", "finalize": "finalize"},
)
graph_builder.add_edge("offer_agent", "finalize")
graph_builder.add_edge("beneficiary_agent", "finalize")
graph_builder.add_edge("init_pdf", END)

# Memory checkpoint
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
thread_id = "demo-thread"

# Display graph visually (requires IPython / Jupyter)
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

# -----------------------------
# Gradio Chat Interface
# -----------------------------

def chat(query: str, history) -> str:
    """Run one chat query through the graph and return the final answer."""
    state = AgentState(messages=[{"role": "user", "content": query}])
    result = graph.invoke(state, config={"configurable": {"thread_id": "demo-thread"}})

    # Access the last message's content via attribute
    last_message = result["messages"][-1]  # this is an AIMessage object
    return last_message.content.replace("\n", "  \n")


# Launch chat interface
gr.ChatInterface(chat).launch()
