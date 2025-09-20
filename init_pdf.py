# app/init_pdf.py
import os
import io
from typing import Tuple, Dict, Any
import fitz  # PyMuPDF
import logging

logger = logging.getLogger("init_pdf")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PDF_PATH = os.getenv("PDF_PATH", os.path.join(DATA_DIR, "Axis-Max-STPP.pdf"))
MAX_PROMPT_CHARS = 28000


def _extract_text_from_file(path: str) -> str:
    try:
        doc = fitz.open(path)
        pages = doc.page_count
        text_pages = []
        for i in range(pages):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            text_pages.append(text)
        full_text = "\n\n".join([f"--- Page {i+1} ---\n{p}" for i, p in enumerate(text_pages)])
        return full_text
    except Exception as e:
        logger.exception("Failed to extract text from PDF path %s: %s", path, e)
        return ""


def init_pdf_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize PDF text into the graph state — but *do not overwrite*
    if `pdf_text` already exists in the incoming state (e.g. from an upload).
    Also ensure we store a preview string in pdf_text (not the tuple).
    """
    incoming_text = state.get("pdf_text") if state else None
    if incoming_text:
        logger.debug("init_pdf_node: pdf_text present in initial state, keeping it.")
        return {"pdf_text": state.get("pdf_text")}
    # otherwise try to load default PDF file if exists
    if os.path.exists(PDF_PATH):
        extracted = _extract_text_from_file(PDF_PATH)
        if len(extracted) > MAX_PROMPT_CHARS:
            extracted = extracted[:MAX_PROMPT_CHARS]
        logger.info("init_pdf_node: loaded default PDF text (%d chars)", len(extracted))
        return {"pdf_text": extracted}
    logger.info("init_pdf_node: no pdf_text provided and default PDF not found.")
    return {}
