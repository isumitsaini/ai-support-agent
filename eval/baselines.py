"""
baselines.py

Two baselines, per assignment requirement ("at least two: a trivial one and a simple one").

BASELINE 1 - "trivial": majority-class classifier + single canned reply + never escalate.
    Predicts the single most common intent in the golden set for every message,
    always returns the same generic reply, and never escalates.
    This establishes the floor: what "doing nothing intelligent" scores.

BASELINE 2 - "simple": keyword/regex intent classifier + template replies per intent
    + rule-based escalation using ONLY the sentiment-intensity heuristic (no LLM
    anywhere in the loop). This is what a team would ship in a day without any
    LLM calls at all. It tells us how much the LLM classification + retrieval +
    generation steps are actually buying us over cheap regex rules - this is the
    comparison that actually matters for justifying the system's complexity/cost.
"""
import json
import re

from taxonomy import INTENTS, INTENT_LIST

# --- Baseline 1: trivial -----------------------------------------------------

def trivial_baseline(customer_text: str) -> dict:
    return {
        "intent": "order_status_delivery",  # most frequent class in golden set
        "confidence": 1.0,
        "draft_reply": "Thanks for reaching out! We're looking into this for you - please DM your order details so we can help further.",
        "escalate": False,
    }


# --- Baseline 2: simple keyword rules -----------------------------------------

KEYWORD_RULES = [
    ("order_status_delivery", r"\b(delivery|deliver|shipped|shipping|tracking|arrive|package|parcel)\b"),
    ("refund_or_return", r"\b(refund|return|money back|reimburse|cashback)\b"),
    ("damaged_or_wrong_item", r"\b(damaged|broken|wrong item|missing item|defective|empty box)\b"),
    ("cancel_order", r"\bcancel\b"),
    ("account_access", r"\b(login|log in|password|account locked|sign in|hack)\b"),
    ("payment_billing", r"\b(charged twice|billing|payment declined|invoice|bank account|unauthorized)\b"),
    ("prime_membership", r"\b(prime|membership|subscription|renew)\b"),
    ("digital_content_technical", r"\b(kindle|app|streaming|prime video|audible|alexa|echo|fire stick)\b"),
    ("service_complaint_escalation", r"\b(worst|terrible|awful|pissed|useless|horrible|appalling|pathetic|sue|lawyer)\b"),
    ("positive_feedback", r"\b(thank you|thanks|appreciate|great service|awesome)\b"),
]

TEMPLATE_REPLIES = {
    "order_status_delivery": "Sorry for the trouble! Please share your order number via DM so we can check the delivery status for you.",
    "refund_or_return": "We're happy to help with that. Please DM your order number so we can look into your refund/return.",
    "damaged_or_wrong_item": "So sorry to hear that! Please DM us photos and your order number so we can make this right.",
    "cancel_order": "We can help with that. Please DM your order number so we can check the cancellation window.",
    "account_access": "For account security we can't assist with login issues here - please use the 'Forgot password' link or DM us to verify your identity.",
    "payment_billing": "We understand this is concerning. Please DM your order number so our billing team can look into this specific charge.",
    "prime_membership": "Happy to help with your Prime membership - please DM any account-specific details.",
    "digital_content_technical": "Sorry for the trouble! Try restarting the app/device first - if that doesn't help, DM us and we'll dig deeper.",
    "service_complaint_escalation": "We're sorry to hear this and want to make it right. Please DM us so we can connect you with the right team.",
    "positive_feedback": "Thank you so much for letting us know - glad we could help!",
    "other": "Thanks for reaching out - could you share a bit more detail so we can help?",
}


def simple_baseline(customer_text: str) -> dict:
    text_lower = customer_text.lower()
    intent = "other"
    for name, pattern in KEYWORD_RULES:
        if re.search(pattern, text_lower):
            intent = name
            break

    # rule-based escalation using ONLY sentiment intensity + taxonomy default flag
    caps_words = re.findall(r"\b[A-Z]{4,}\b", customer_text)
    repeated_punct = re.search(r"[!?]{2,}", customer_text)
    high_intensity = len(caps_words) >= 2 or bool(repeated_punct)

    default_escalate = INTENTS.get(intent, {}).get("default_escalate", True)
    escalate = default_escalate or high_intensity

    return {
        "intent": intent,
        "confidence": 0.5,  # baseline doesn't produce a real confidence score
        "draft_reply": TEMPLATE_REPLIES.get(intent, TEMPLATE_REPLIES["other"]),
        "escalate": escalate,
    }


if __name__ == "__main__":
    # quick smoke test against golden set (classification + escalation agreement only,
    # no reply quality judged here - that's eval/run_eval.py's job)
    golden = [json.loads(l) for l in open("data/golden/golden_set.jsonl")]

    for name, fn in [("trivial", trivial_baseline), ("simple", simple_baseline)]:
        correct_intent = 0
        correct_escalate = 0
        for row in golden:
            pred = fn(row["customer_text"])
            if pred["intent"] == row["gold_intent"]:
                correct_intent += 1
            if pred["escalate"] == row["gold_should_escalate"]:
                correct_escalate += 1
        n = len(golden)
        print(f"{name:10s} intent_acc={correct_intent/n:.3f}  escalation_acc={correct_escalate/n:.3f}")
