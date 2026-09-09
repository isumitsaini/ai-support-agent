"""
judge.py

LLM-as-judge for reply quality. Scores a candidate draft reply against 4 rubric
dimensions, each 1-5, given the customer message and (when available) grounding
examples of how the brand has actually replied to similar issues historically.

RUBRIC (documented so a human re-grading can apply it identically):
  1. RELEVANCE     - Does the reply actually address what the customer said, not
                      a generic non-sequitur? (1=off-topic, 5=directly on point)
  2. GROUNDEDNESS  - Does the reply stay consistent with brand policy/precedent
                      (doesn't invent a refund amount, doesn't promise something
                      the brand doesn't do)? (1=fabricates/contradicts precedent, 5=fully consistent)
  3. TONE          - Appropriately empathetic without being saccharine, matches
                      brand voice (brief, professional, not robotic). (1=tone-deaf, 5=well-calibrated)
  4. ACTIONABILITY - Gives the customer a clear next step (DM link, specific
                      action) rather than a vague platitude. (1=no next step, 5=clear next step)

Overall score = mean of the 4 dimensions (not a 5th separately-judged number) -
this keeps the judge's output decomposable for failure analysis rather than a
single opaque number (see report "what's misleading about my headline number").

We do NOT ask the judge to compare against the brand's actual historical reply
directly as a gold standard, because the historical reply is itself noisy
(sometimes curt, sometimes wrong) - see decision log D12. Instead the judge
scores the candidate reply on its own merits against the rubric.
"""
import json
import re

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

JUDGE_SYSTEM = """You are an expert customer support quality auditor for Amazon's Twitter
support account. Score the CANDIDATE REPLY against this rubric, each dimension 1-5:

1. relevance: Does the reply address what the customer actually said?
2. groundedness: Is the reply consistent with realistic brand policy (no invented
   promises, refund amounts, or dates not evidenced)?
3. tone: Appropriately empathetic, brief, professional - not robotic, not saccharine.
4. actionability: Does it give the customer a clear, concrete next step?

Return ONLY JSON:
{"relevance": <1-5>, "groundedness": <1-5>, "tone": <1-5>, "actionability": <1-5>, "rationale": "<one sentence>"}
No markdown fences, no other text."""


def judge_reply(customer_text: str, candidate_reply: str, intent: str = None) -> dict:
    user_content = (
        f"Customer message: {customer_text}\n\n"
        f"Classified intent: {intent or 'unknown'}\n\n"
        f"Candidate reply to score: {candidate_reply}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    text = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"relevance": None, "groundedness": None, "tone": None, "actionability": None, "rationale": "parse_failure"}

    dims = ["relevance", "groundedness", "tone", "actionability"]
    scores = [result.get(d) for d in dims if isinstance(result.get(d), (int, float))]
    result["overall"] = sum(scores) / len(scores) if scores else None
    return result


if __name__ == "__main__":
    # smoke test structure only - requires ANTHROPIC_API_KEY to actually run
    example = judge_reply(
        customer_text="my package says delivered but I never got it",
        candidate_reply="So sorry about that! Please DM your order number and we'll help track it down.",
        intent="order_status_delivery",
    )
    print(json.dumps(example, indent=2))
