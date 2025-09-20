# app/agents/provider_agent.py
import time
import random
import uuid
from typing import Dict, Any, List
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

def generate_providers(n:int=5, seed_text: str = "") -> List[Dict[str, Any]]:
    rand = random.Random()
    rand.seed(seed_text + str(random.random()))
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

def provider_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent node to produce provider candidates based on the latest user message.
    Returns partial state dict with:
      - provider_candidates: list
      - provider_last_updated: timestamp
      - messages: assistant message (tagged with agent: "provider")
    """
    user_query = get_last_human_message(state)

    try:
        candidates = generate_providers(n=6, seed_text=user_query)
    except Exception as e:
        logger.exception("provider generation failed")
        return {"messages": [{"role": "assistant", "content": f"Provider generation failed: {e}", "agent": "provider"}]}

    return {
        "provider_candidates": candidates,
        "provider_last_updated": int(time.time()),
        "messages": [
            {
                "role": "assistant",
                "content": f"Provider Agent found {len(candidates)} candidates.",
                "agent": "provider",          # tag so finalize can detect who last ran
                "response_key": "provider_candidates"
            }
        ],
    }
