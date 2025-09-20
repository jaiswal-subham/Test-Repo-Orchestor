# app/agents/provider_agent.py
import time
import random
import uuid
from typing import Dict, Any, List, Optional
import logging

from utility import get_last_human_message

logger = logging.getLogger("provider_agent")

# Pools (use your original lists)
FIRST = ["Aarav", "Ishita", "Rohan", "Priya", "Samar", "Nina", "Arjun", "Maya", "Vikram", "Kavya"]
LAST = ["Sharma", "Patel", "Singh", "Gupta", "Rao", "Desai", "Mehra", "Chatterjee"]
SPECIALTIES = ["Cardiology", "Dermatology", "Orthopedics", "Pediatrics", "ENT", "Psychiatry", "General Medicine", "Ophthalmology"]
GENDERS = ["Male", "Female", "Non-binary"]
LOCATIONS = ["Bhubaneswar", "Bengaluru", "Delhi", "Mumbai", "Hyderabad", "Kolkata", "Pune"]


def _random_email(name: str) -> str:
    slug = name.lower().replace(" ", ".")
    domain = random.choice(["example.com", "clinicmail.com", "provider.org"])
    return f"{slug}@{domain}"


def generate_providers(n: int = 5, seed_text: str = "") -> List[Dict[str, Any]]:
    rand = random.Random()
    rand.seed((seed_text or "") + str(random.random()))
    providers = []
    for _ in range(n):
        name = f"{rand.choice(FIRST)} {rand.choice(LAST)}"
        p = {
            "id": str(uuid.uuid4()),
            "name": name,
            "gender": rand.choice(GENDERS),
            "specialty": rand.choice(SPECIALTIES),
            "rating": round(rand.uniform(3.5, 5.0), 1),
            "years_experience": rand.randint(2, 30),
            "location": rand.choice(LOCATIONS),
            "contact_email": _random_email(name),
        }
        providers.append(p)
    return providers


def _make_final_answer_from_providers(candidates: List[Dict[str, Any]]) -> str:
    """
    Produce a concise human-readable textual content summarizing providers.
    Example: "Found 6 providers: Dr A X (Cardiology), Dr B Y (Dermatology), ..."
    Limit to top 5 names to keep final_answer short.
    """
    if not candidates:
        return "No providers found."
    count = len(candidates)
    top = candidates[:5]
    name_parts = []
    for p in top:
        name = p.get("name", "Unknown")
        spec = p.get("specialty")
        if spec:
            name_parts.append(f"{name} ({spec})")
        else:
            name_parts.append(name)
    names_summary = ", ".join(name_parts)
    more = f" and {count - len(top)} more" if count > len(top) else ""
    return f"Found {count} providers: {names_summary}{more}."


def provider_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce provider candidates and write a textual final_answer (not the whole object).
    Writes:
      - provider_candidates: the list (raw)
      - provider_last_updated: timestamp
      - final_answer: textual summary produced by _make_final_answer_from_providers
      - messages: assistant message with content == final_answer
    """
    user_query: Optional[str] = get_last_human_message(state)

    try:
        candidates = generate_providers(n=6, seed_text=user_query or "")
    except Exception as e:
        logger.exception("provider generation failed")
        err_msg = f"Provider generation failed: {e}"
        return {
            "messages": [
                {"role": "assistant", "content": err_msg, "agent": "provider", "response_key": "provider_candidates"}
            ],
            "final_answer": err_msg,
        }

    final_answer = _make_final_answer_from_providers(candidates)

    return {
        "provider_candidates": candidates,
        "provider_last_updated": int(time.time()),
        "final_answer": final_answer,
        "messages": [
            {
                "role": "assistant",
                "content": final_answer,
                "agent": "provider",
                "response_key": "provider_candidates",
            }
        ],
    }
