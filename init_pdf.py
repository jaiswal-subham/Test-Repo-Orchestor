# app/agents/init_pdf.py
import os, io
from typing import Tuple, Dict, Any
import fitz  # PyMuPDF
import logging

logger = logging.getLogger("init_pdf")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
PDF_PATH = os.getenv("PDF_PATH", os.path.join(DATA_DIR, "Axis-Max-STPP.pdf"))
MAX_PROMPT_CHARS = 28000

# in app/agents/init_pdf.py

def init_pdf_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize PDF text into the graph state — but *do not overwrite*
    if `pdf_text` already exists in the incoming state (e.g. from an upload).
    Also ensure we store a preview string in pdf_text (not the tuple).
    """
    # If incoming state already contains pdf_text (provided by the upload),
    # generate a preview if necessary but DO NOT overwrite the original full text.
    incoming_text = state.get("pdf_text") if state else None
    if incoming_text:
        logger.debug("init_pdf_node: pdf_text present in initial state, preparing preview.")
        # If the incoming value is already a tuple (preview, bool) handle safely:
        return {"pdf_text": state.get("pdf_text")}
