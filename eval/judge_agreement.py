"""
judge_agreement.py

Measures how well the LLM judge (judge.py) agrees with a human rater, per the
assignment's explicit requirement: "evidence of how well your judge agrees
with a human."

METHOD:
  1. Take a 40-example subsample of (customer_text, candidate_reply) pairs -
     a mix of real historical brand replies AND deliberately-injected bad
     replies (wrong intent, fabricated promises, tone-deaf, no next step) so
     the agreement check spans the full quality range rather than only
     clustering near "pretty good" (see decision log D13 - if every example
     scores 4-5, agreement looks artificially high because there's no
     disagreement to measure).
  2. A human (me) independently scores the same 40 examples on the same 4-dim
     rubric, blind to the judge's scores at scoring time.
  3. Report: exact agreement rate (same 1-5 score), agreement-within-1 rate,
     and correlation, per dimension AND on the averaged overall score.

We report agreement-within-1 as the headline number because 1-5 Likert scales
have well-known "off-by-one" noise even between two humans; treating that as
total disagreement would understate real judge quality. But we still show
exact-match too, since collapsing to only within-1 would look artificially
strong (this exact tension is called out in report Section "misleading about
my headline number").
"""
import json
import statistics

# 40 hand-constructed (customer_text, candidate_reply, human_scores) examples
# spanning good/bad replies. human_scores = {relevance, groundedness, tone, actionability}
# These are deliberately diverse in quality to make agreement measurement meaningful.
HUMAN_LABELED_REPLIES = [
    {
        "customer_text": "my package says delivered but I never got it",
        "candidate_reply": "So sorry about that! Please DM your order number and we'll help track it down.",
        "human_scores": {"relevance": 5, "groundedness": 5, "tone": 5, "actionability": 5},
    },
    {
        "customer_text": "my package says delivered but I never got it",
        "candidate_reply": "Thanks for your patience, we're always here to help with anything Amazon related!",
        "human_scores": {"relevance": 1, "groundedness": 3, "tone": 3, "actionability": 1},
    },
    {
        "customer_text": "I want a refund for my damaged item, order #123-4567",
        "candidate_reply": "We've processed a full refund of $49.99 to your account, it will arrive in 2 days.",
        "human_scores": {"relevance": 4, "groundedness": 1, "tone": 4, "actionability": 3},
    },
    {
        "customer_text": "I want a refund for my damaged item, order #123-4567",
        "candidate_reply": "So sorry to hear that! Please DM us photos of the damage and your order number so we can process a refund or replacement.",
        "human_scores": {"relevance": 5, "groundedness": 5, "tone": 5, "actionability": 5},
    },
    {
        "customer_text": "how do I cancel my prime membership",
        "candidate_reply": "You can manage your Prime membership anytime from Account > Prime Membership > End Membership.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 5},
    },
    {
        "customer_text": "how do I cancel my prime membership",
        "candidate_reply": "We're sorry you're having trouble! Please try restarting the app.",
        "human_scores": {"relevance": 1, "groundedness": 2, "tone": 2, "actionability": 1},
    },
    {
        "customer_text": "this is the third time I've called about my missing refund, nobody helps",
        "candidate_reply": "I understand your frustration - let's get this sorted. Please DM your order number and case reference so a specialist can prioritize this.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 5, "actionability": 5},
    },
    {
        "customer_text": "this is the third time I've called about my missing refund, nobody helps",
        "candidate_reply": "Please DM your order number so we can help!",
        "human_scores": {"relevance": 3, "groundedness": 4, "tone": 2, "actionability": 4},
    },
    {
        "customer_text": "my kindle won't turn on, already tried charging it overnight",
        "candidate_reply": "Try plugging it in for at least 30 minutes using the original cable and adapter, then hold the power button for 40 seconds.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 3, "actionability": 5},
    },
    {
        "customer_text": "my kindle won't turn on, already tried charging it overnight",
        "candidate_reply": "Have you tried turning it off and on again?",
        "human_scores": {"relevance": 2, "groundedness": 3, "tone": 2, "actionability": 2},
    },
    {
        "customer_text": "someone hacked my account and changed my email, I need this fixed now",
        "candidate_reply": "For your security, please reset your password using the link on the sign-in page and contact us via DM to verify your identity so we can secure the account.",
        "human_scores": {"relevance": 5, "groundedness": 5, "tone": 4, "actionability": 5},
    },
    {
        "customer_text": "someone hacked my account and changed my email, I need this fixed now",
        "candidate_reply": "Please share your current password so we can verify your identity.",
        "human_scores": {"relevance": 3, "groundedness": 1, "tone": 2, "actionability": 3},
    },
    {
        "customer_text": "thanks so much for your help, issue resolved!",
        "candidate_reply": "You're so welcome! Glad we could help - have a great day!",
        "human_scores": {"relevance": 5, "groundedness": 5, "tone": 5, "actionability": 5},
    },
    {
        "customer_text": "thanks so much for your help, issue resolved!",
        "candidate_reply": "We're sorry to hear that! Please DM your order number so we can look into it.",
        "human_scores": {"relevance": 1, "groundedness": 2, "tone": 1, "actionability": 1},
    },
    {
        "customer_text": "your delivery driver left my package in the rain, it's ruined",
        "candidate_reply": "Oh no, I'm sorry to hear that! Please DM us photos of the damage and your order number so we can send a replacement or refund.",
        "human_scores": {"relevance": 5, "groundedness": 5, "tone": 5, "actionability": 5},
    },
    {
        "customer_text": "your delivery driver left my package in the rain, it's ruined",
        "candidate_reply": "We appreciate your feedback and are always working to improve.",
        "human_scores": {"relevance": 1, "groundedness": 3, "tone": 2, "actionability": 1},
    },
    {
        "customer_text": "I was charged twice for the same order, please fix this",
        "candidate_reply": "I've refunded the duplicate charge, you'll see it in 3-5 business days.",
        "human_scores": {"relevance": 5, "groundedness": 1, "tone": 4, "actionability": 3},
    },
    {
        "customer_text": "I was charged twice for the same order, please fix this",
        "candidate_reply": "So sorry about that! Please DM your order number so our billing team can verify and correct the duplicate charge.",
        "human_scores": {"relevance": 5, "groundedness": 5, "tone": 5, "actionability": 5},
    },
    {
        "customer_text": "when will the new echo dot be back in stock",
        "candidate_reply": "We don't have a specific restock date, but you can turn on 'notify me' on the product page to get an alert when it's available.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 4},
    },
    {
        "customer_text": "when will the new echo dot be back in stock",
        "candidate_reply": "So sorry for the inconvenience, please DM your order number.",
        "human_scores": {"relevance": 1, "groundedness": 2, "tone": 2, "actionability": 2},
    },
    {
        "customer_text": "your app keeps crashing every time I open prime video",
        "candidate_reply": "Sorry about that! Try uninstalling and reinstalling the app, and make sure it's updated to the latest version. If it still crashes, DM us your device model.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 5},
    },
    {
        "customer_text": "your app keeps crashing every time I open prime video",
        "candidate_reply": "We're glad you're enjoying Prime Video!",
        "human_scores": {"relevance": 1, "groundedness": 2, "tone": 1, "actionability": 1},
    },
    {
        "customer_text": "I never received a confirmation email for my order",
        "candidate_reply": "Please check your spam folder first - if it's not there, DM your order number and we'll resend confirmation.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 5},
    },
    {
        "customer_text": "I never received a confirmation email for my order",
        "candidate_reply": "Your order has been cancelled as requested.",
        "human_scores": {"relevance": 1, "groundedness": 1, "tone": 2, "actionability": 1},
    },
    {
        "customer_text": "can I return an item I bought 3 months ago",
        "candidate_reply": "Standard returns are within 30 days, but DM your order number and we can check if an exception applies to your case.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 5},
    },
    {
        "customer_text": "can I return an item I bought 3 months ago",
        "candidate_reply": "Yes! You can return it anytime within a year for a full refund, no questions asked.",
        "human_scores": {"relevance": 4, "groundedness": 1, "tone": 4, "actionability": 3},
    },
    {
        "customer_text": "worst customer service ever, I've been on hold for an hour",
        "candidate_reply": "I completely understand your frustration and I'm sorry for the wait. Please DM your details so a specialist can reach out directly.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 5, "actionability": 4},
    },
    {
        "customer_text": "worst customer service ever, I've been on hold for an hour",
        "candidate_reply": "Try restarting your router and calling again.",
        "human_scores": {"relevance": 1, "groundedness": 2, "tone": 1, "actionability": 2},
    },
    {
        "customer_text": "does the fire stick work with a hotel wifi that needs a login page",
        "candidate_reply": "Fire TV Stick can have trouble with wifi networks requiring a browser login. Try connecting a phone hotspot as a workaround, or DM us for more troubleshooting steps.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 4},
    },
    {
        "customer_text": "does the fire stick work with a hotel wifi that needs a login page",
        "candidate_reply": "Yes, all our devices are fully compatible with any network configuration.",
        "human_scores": {"relevance": 3, "groundedness": 1, "tone": 3, "actionability": 1},
    },
    {
        "customer_text": "I want to speak to a manager about my cancelled order",
        "candidate_reply": "I hear you, and I'm sorry for the frustration. Please DM your order number and we'll escalate this to a supervisor.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 5, "actionability": 5},
    },
    {
        "customer_text": "I want to speak to a manager about my cancelled order",
        "candidate_reply": "Your order has been successfully placed!",
        "human_scores": {"relevance": 1, "groundedness": 1, "tone": 1, "actionability": 1},
    },
    {
        "customer_text": "the item I received doesn't match the photos at all",
        "candidate_reply": "That's not right - please DM photos of what you received along with your order number so we can make this right with a refund or replacement.",
        "human_scores": {"relevance": 5, "groundedness": 5, "tone": 5, "actionability": 5},
    },
    {
        "customer_text": "the item I received doesn't match the photos at all",
        "candidate_reply": "Our photos are professionally taken and accurately represent the product.",
        "human_scores": {"relevance": 2, "groundedness": 2, "tone": 1, "actionability": 1},
    },
    {
        "customer_text": "is same-day delivery available in rural areas",
        "candidate_reply": "Same-day delivery availability depends on your specific postal code - you can check by entering your address at checkout.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 4},
    },
    {
        "customer_text": "is same-day delivery available in rural areas",
        "candidate_reply": "So sorry for the inconvenience this has caused!",
        "human_scores": {"relevance": 1, "groundedness": 2, "tone": 1, "actionability": 1},
    },
    {
        "customer_text": "my alexa stopped responding to my voice commands today",
        "candidate_reply": "Sorry about that! Try saying 'Alexa' a bit louder and check the mic is not muted (ring light should be blue, not red). If that doesn't work, DM us.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 5},
    },
    {
        "candidate_reply": "We've issued you a $500 gift card as an apology.",
        "customer_text": "my alexa stopped responding to my voice commands today",
        "human_scores": {"relevance": 1, "groundedness": 1, "tone": 2, "actionability": 2},
    },
    {
        "customer_text": "why did my order get automatically cancelled at checkout",
        "candidate_reply": "This can happen due to payment verification issues or item availability changes. DM your order number and we can look into the specific reason.",
        "human_scores": {"relevance": 5, "groundedness": 4, "tone": 4, "actionability": 5},
    },
    {
        "customer_text": "why did my order get automatically cancelled at checkout",
        "candidate_reply": "Congratulations on your recent purchase!",
        "human_scores": {"relevance": 1, "groundedness": 1, "tone": 1, "actionability": 1},
    },
]


def pearson_corr(x, y):
    if len(x) < 2 or len(set(x)) == 1 or len(set(y)) == 1:
        return None
    return statistics.correlation(x, y)


def run_agreement_check(judge_fn):
    """judge_fn: callable(customer_text, candidate_reply) -> dict with 4 dims + overall"""
    dims = ["relevance", "groundedness", "tone", "actionability"]
    per_dim_exact = {d: 0 for d in dims}
    per_dim_within1 = {d: 0 for d in dims}
    per_dim_human_scores = {d: [] for d in dims}
    per_dim_judge_scores = {d: [] for d in dims}

    n = len(HUMAN_LABELED_REPLIES)
    results = []
    for ex in HUMAN_LABELED_REPLIES:
        judge_result = judge_fn(ex["customer_text"], ex["candidate_reply"])
        row = {"customer_text": ex["customer_text"], "candidate_reply": ex["candidate_reply"],
               "human": ex["human_scores"], "judge": judge_result}
        results.append(row)
        for d in dims:
            h, j = ex["human_scores"][d], judge_result.get(d)
            if j is None:
                continue
            per_dim_human_scores[d].append(h)
            per_dim_judge_scores[d].append(j)
            if h == j:
                per_dim_exact[d] += 1
            if abs(h - j) <= 1:
                per_dim_within1[d] += 1

    print(f"{'dimension':16s} {'exact%':>8s} {'within1%':>10s} {'pearson_r':>10s}")
    for d in dims:
        exact_pct = per_dim_exact[d] / n * 100
        within1_pct = per_dim_within1[d] / n * 100
        r = pearson_corr(per_dim_human_scores[d], per_dim_judge_scores[d])
        r_str = f"{r:.3f}" if r is not None else "n/a"
        print(f"{d:16s} {exact_pct:7.1f}% {within1_pct:9.1f}% {r_str:>10s}")

    # overall (averaged) score agreement
    human_overall = [sum(ex["human_scores"].values()) / 4 for ex in HUMAN_LABELED_REPLIES]
    judge_overall = [r["judge"].get("overall") for r in results]
    valid_pairs = [(h, j) for h, j in zip(human_overall, judge_overall) if j is not None]
    if valid_pairs:
        hs, js = zip(*valid_pairs)
        r = pearson_corr(list(hs), list(js))
        within1 = sum(1 for h, j in valid_pairs if abs(h - j) <= 1) / len(valid_pairs) * 100
        print(f"\n{'overall (avg)':16s} {'':>8s} {within1:9.1f}% {r:>10.3f}" if r else f"\n{'overall (avg)':16s}")

    return results


if __name__ == "__main__":
    from judge import judge_reply
    run_agreement_check(lambda text, reply: judge_reply(text, reply))
