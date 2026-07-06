import os
import json
from anthropic import Anthropic
from app.crux_engine import Claim

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

ADVOCATE_PROMPT = """You are a rigorous debate agent assigned to build the strongest possible case IN FAVOR of the following claim.

Your response must be a JSON object with this exact shape:
{
  "claims": [
    {
      "id": "a1",
      "text": "A single, specific, falsifiable claim.",
      "claim_type": "empirical" | "causal" | "value" | "definitional",
      "depends_on": "a2" or ""
    }
  ]
}

Rules:
- Write exactly 4 claims
- Each claim must be one sentence, specific, and falsifiable
- depends_on must reference another claim id in your list that this claim builds on, or empty string if it is a root claim
- Return only valid JSON, no preamble, no markdown
- claim_type must be one of: empirical, causal, value, definitional

Claim to argue for: {topic}"""

SKEPTIC_PROMPT = """You are a rigorous debate agent assigned to build the strongest possible case AGAINST the following claim.

Your response must be a JSON object with this exact shape:
{
  "claims": [
    {
      "id": "s1",
      "text": "A single, specific, falsifiable claim.",
      "claim_type": "empirical" | "causal" | "value" | "definitional",
      "depends_on": "s2" or ""
    }
  ]
}

Rules:
- Write exactly 4 claims
- Each claim must be one sentence, specific, and falsifiable
- depends_on must reference another claim id in your list that this claim builds on, or empty string if it is a root claim
- Return only valid JSON, no preamble, no markdown
- claim_type must be one of: empirical, causal, value, definitional

Claim to argue against: {topic}"""

def _parse_claims(raw: str, agent: str) -> list[Claim]:
    data = json.loads(raw)
    return[
        Claim(
            id=c["id"],
            text=c["text"],
            agent=agent,
            claim_type=c["claim_type"],
            depends_on=c.get("depends_on", ""),

        )
        for c in data["claims"]
    ]

def generate_arguments(topic: str) -> tuple[list[Claim], list[Claim]]:
    if DRY_RUN:
        print("[DRY RUN] Skipping Claude API calls")
        advocate_claims = [
            Claim("a1", "The evidence strongly supports this position.", "advocate", "empirical", ""),
            Claim("a2", "Studies confirm the causal relationship.", "advocate", "causal", "a1"),
            Claim("a3", "The sample size is sufficient for generalization.", "advocate", "empirical", "a1"),
            Claim("a4", "Expert consensus aligns with this view.", "advocate", "empirical", "a2"),
        ]
        skeptic_claims = [
            Claim("s1", "The evidence has significant methodological flaws.", "skeptic", "empirical", ""),
            Claim("s2", "Correlation does not establish causation here.", "skeptic", "causal", "s1"),
            Claim("s3", "The sample is too narrow to generalize.", "skeptic", "empirical", "s1"),
            Claim("s4", "Expert opinion is divided on this topic.", "skeptic", "empirical", "s2"),
        ]
        return advocate_claims, skeptic_claims
    
    print(f"Calling Claude (advocate)...")
    advocate_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": ADVOCATE_PROMPT.format(topic=topic)}],
    )

    print(f"Calling Claude (skeptic)...")
    skeptic_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": SKEPTIC_PROMPT.format(topic=topic)}]

    )

    advocate_claims = _parse_claims(advocate_response.content[0].text, "advocate")
    skeptic_claims = _parse_claims(skeptic_response.content[0].text, "skeptic")

    return advocate_claims, skeptic_claims