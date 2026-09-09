"""
agent.py

The support agent pipeline: for one incoming customer message, produce
  {intent, confidence, draft_reply, escalate (bool), escalation_reason}

Pipeline (see report Section "System Design"):
  1. CLASSIFY   - LLM call, constrained to the taxonomy, returns intent + confidence (self-reported).
  2. RETRIEVE   - pull top-3 historically similar *same-intent* resolutions from the
                  TF-IDF index (retrieval.py) to ground the draft in real precedent.
  3. DRAFT      - LLM call conditioned on the retrieved precedent + brand voice guide.
  4. ESCALATE   - rule-based policy layer (NOT another LLM call) that combines:
                    - taxonomy default_escalate flag
                    - classifier confidence threshold
                    - simple safety triggers (PII/credentials request, legal/safety keywords)
                    - sentiment intensity heuristic (ALL CAPS, repeated punctuation, "lawyer",
                      "sue", "unsafe", "allergic", "child", etc.)
                  This is deliberately NOT delegated to the LLM: escalation is a safety-critical
                  binary decision and we want it auditable, deterministic, and testable in
                  isolation from generation quality (see decision log D9, D10).

Design note: classification and drafting are DELIBERATELY separate LLM calls rather than
one call that returns everything. This lets us (a) evaluate classification accuracy in
isolation against the golden set, (b) cache/reuse classification for retrieval filtering
before spending tokens on a draft, and (c) unit test the escalation policy without needing
an LLM in the loop at all (see eval/test_escalation_policy.py).
"""
import json
import re
from dataclasses import dataclass, field

import anthropic

from taxonomy import INTENTS, INTENT_LIST
from retrieval import ResolutionRetriever

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

CLASSIFY_SYSTEM = f"""Classify the customer support message into exactly one intent from:
{json.dumps(INTENT_LIST)}

Intent definitions:
{json.dumps({k: v['description'] for k, v in INTENTS.items()}, indent=2)}

Return ONLY JSON: {{"intent": "<one of the list>", "confidence": <0.0-1.0>, "rationale": "<one short sentence>"}}
No markdown fences, no other text."""

DRAFT_SYSTEM = """You are drafting a Twitter customer support reply for Amazon (@AmazonHelp),
in their established voice: brief, empathetic, never over-promises, never asks for
sensitive info (passwords, full card numbers) in a public reply, and directs the
customer to DM or a help link for anything requiring account-specific lookup.

You are given real historical examples of how this brand has resolved similar issues.
Use them as a STYLE AND POLICY guide, not a template to copy verbatim - adapt to this
specific customer's message. Keep it under 280 characters. Do not invent order numbers,
refund amounts, or policies not evidenced in the examples.

Return ONLY JSON: {"draft_reply": "<the reply text>"}
No markdown fences, no other text."""

# --- Escalation policy (deterministic, auditable) ---------------------------

SAFETY_TRIGGER_PATTERNS = [
    (r"\b(lawyer|sue|legal action|attorney)\b", "legal threat language detected"),
    (r"\b(allerg(y|ic)|hospital|injur(y|ed)|unsafe|fire hazard|explod)\b", "potential safety/health incident"),
    (r"\b(child|kid|minor)\b.*\b(hurt|injur|unsafe)\b", "safety incident involving a minor"),
    (r"\b(fraud|unauthorized charge|stolen|hacked|identity theft)\b", "potential fraud/security incident"),
    (r"\b(password|social security|ssn|full card number|cvv)\b", "sensitive credential/PII mentioned"),
]

CONFIDENCE_ESCALATION_THRESHOLD = 0.55


def _detect_safety_triggers(text: str):
    hits = []
    for pattern, reason in SAFETY_TRIGGER_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(reason)
    return hits


def _sentiment_intensity_flag(text: str) -> bool:
    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    repeated_punct = re.search(r"[!?]{2,}", text)
    return len(caps_words) >= 2 or bool(repeated_punct)


@dataclass
class EscalationDecision:
    escalate: bool
    reasons: list = field(default_factory=list)


def decide_escalation(customer_text: str, intent: str, confidence: float) -> EscalationDecision:
    reasons = []

    safety_hits = _detect_safety_triggers(customer_text)
    if safety_hits:
        reasons.extend(safety_hits)

    if intent not in INTENTS:
        reasons.append(f"unknown intent label '{intent}' - fail safe to human")
    elif INTENTS[intent]["default_escalate"]:
        reasons.append(f"intent '{intent}' is policy-flagged for human handling")

    if confidence < CONFIDENCE_ESCALATION_THRESHOLD:
        reasons.append(f"classifier confidence {confidence:.2f} below threshold {CONFIDENCE_ESCALATION_THRESHOLD}")

    if _sentiment_intensity_flag(customer_text) and intent == "service_complaint_escalation":
        reasons.append("high sentiment intensity compounding an already-flagged complaint intent")

    return EscalationDecision(escalate=len(reasons) > 0, reasons=reasons)


# --- LLM steps ----------------------------------------------------------------

def _parse_json_response(text: str) -> dict:
    text = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def classify_intent(customer_text: str) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": customer_text}],
    )
    try:
        result = _parse_json_response(resp.content[0].text)
    except (json.JSONDecodeError, IndexError):
        result = {"intent": "other", "confidence": 0.0, "rationale": "parse_failure_fallback"}
    if result.get("intent") not in INTENT_LIST:
        result["intent"] = "other"
    result["confidence"] = float(result.get("confidence", 0.0))
    return result


def draft_reply(customer_text: str, intent: str, retrieved_examples: list) -> str:
    examples_block = "\n\n".join(
        f"Example {i+1} (similar {ex['intent']} case):\n"
        f"  Customer: {ex['customer_text']}\n"
        f"  Brand replied: {ex['brand_reply']}"
        for i, ex in enumerate(retrieved_examples)
    ) or "(No close historical example found - use general brand voice policy above.)"

    user_content = (
        f"Customer message to reply to:\n{customer_text}\n\n"
        f"Classified intent: {intent}\n\n"
        f"Historical precedent:\n{examples_block}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=DRAFT_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    try:
        result = _parse_json_response(resp.content[0].text)
        return result.get("draft_reply", "").strip()
    except (json.JSONDecodeError, IndexError):
        return resp.content[0].text.strip()  # fall back to raw text rather than empty


# --- Full pipeline --------------------------------------------------------

def run_agent(customer_text: str, retriever: ResolutionRetriever = None) -> dict:
    cls = classify_intent(customer_text)
    intent, confidence = cls["intent"], cls["confidence"]

    examples = []
    if retriever is not None:
        examples = retriever.query(customer_text, intent=intent, k=3)

    reply = draft_reply(customer_text, intent, examples)

    esc = decide_escalation(customer_text, intent, confidence)

    return {
        "customer_text": customer_text,
        "intent": intent,
        "confidence": confidence,
        "classification_rationale": cls.get("rationale", ""),
        "retrieved_examples": examples,
        "draft_reply": reply,
        "escalate": esc.escalate,
        "escalation_reasons": esc.reasons,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--index", default="data/processed/retrieval_index.pkl")
    args = ap.parse_args()

    retriever = ResolutionRetriever.load(args.index)
    result = run_agent(args.message, retriever)
    print(json.dumps(result, indent=2, ensure_ascii=False))
