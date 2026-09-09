"""
taxonomy.py

Intent taxonomy for AmazonHelp, derived empirically from keyword frequency analysis
over 15,000 sampled customer->brand pairs (see notebooks/01_intent_exploration.md
for the frequency table this was built from).

Design decisions (see decision log D4-D6):
- 9 intents + "other". We deliberately did NOT try to match Amazon's real internal
  taxonomy (unknown to us, likely 50-100+ categories) - we built the smallest set
  that (a) covers >90% of real volume in our sample and (b) has genuinely different
  resolution and escalation behavior per class. Two intents that "sound" different
  but get identical treatment were merged.
- "other" is a first-class intent, not a leftover bucket - see report Section 1.
"""

INTENTS = {
    "order_status_delivery": {
        "description": "Customer asking where their order/package is, delivery timing, tracking issues, or delivery marked complete but not received.",
        "typical_resolution": "Ask for order number via DM/link, point to tracking, apologize for delay, offer to escalate to carrier if truly lost.",
        "default_escalate": False,
    },
    "refund_or_return": {
        "description": "Customer wants a refund, is asking about return status, or says they were charged incorrectly for a return.",
        "typical_resolution": "Direct to Returns Center, explain refund timelines, request order # for specific case lookup.",
        "default_escalate": False,
    },
    "damaged_or_wrong_item": {
        "description": "Item arrived damaged, defective, or the wrong item/missing items in the package.",
        "typical_resolution": "Apologize, request photos/order # via DM, offer replacement or refund path.",
        "default_escalate": False,
    },
    "cancel_order": {
        "description": "Customer wants to cancel an order or subscription before it ships/renews.",
        "typical_resolution": "Explain cancellation window, direct to Orders page or DM for manual cancellation if past window.",
        "default_escalate": False,
    },
    "account_access": {
        "description": "Login failures, password reset, account locked, 2FA/verification code issues.",
        "typical_resolution": "Never request credentials in public reply; direct to password reset flow or DM for identity verification.",
        "default_escalate": True,  # security-sensitive - human/secure-channel handoff by policy
    },
    "payment_billing": {
        "description": "Double charges, declined payments, invoice/billing discrepancies not tied to a specific return.",
        "typical_resolution": "Apologize, request order # via DM, explain that billing team needs to verify specific transaction.",
        "default_escalate": True,  # financial dispute, needs verified human review
    },
    "prime_membership": {
        "description": "Questions or complaints about Prime membership, renewal, benefits, pricing.",
        "typical_resolution": "Explain membership management via account page, offer to look into unexpected renewal charges via DM.",
        "default_escalate": False,
    },
    "digital_content_technical": {
        "description": "Kindle, Prime Video, app, or download/streaming technical issues.",
        "typical_resolution": "Basic troubleshooting (restart app/device), link to help article, escalate if says troubleshooting already tried.",
        "default_escalate": False,
    },
    "service_complaint_escalation": {
        "description": "Strong negative sentiment about service quality itself (not a specific transactional issue) - e.g. repeated bad experiences, threats to leave, already-failed prior contact.",
        "typical_resolution": "De-escalate tone, acknowledge frustration, offer direct escalation path; do NOT attempt to resolve with generic troubleshooting.",
        "default_escalate": True,  # sentiment + prior-failed-contact signal
    },
    "positive_feedback": {
        "description": "Thanks, praise, or a resolved-issue acknowledgment - no action needed.",
        "typical_resolution": "Brief thank-you acknowledgment.",
        "default_escalate": False,
    },
    "other": {
        "description": "Doesn't fit the above - ambiguous, off-topic, or a fragment lacking enough context to classify confidently.",
        "typical_resolution": "N/A - route to human by default; do not force a generic reply onto content the model doesn't understand.",
        "default_escalate": True,  # low-confidence catch-all escalates by policy, not by content
    },
}

INTENT_LIST = list(INTENTS.keys())
