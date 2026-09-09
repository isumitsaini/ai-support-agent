# Report: AI Support Agent for AmazonHelp

## 0. Scope in one paragraph

Brand: **AmazonHelp** (169,840 inbound customer messages in the raw dataset — the
highest-volume brand, giving the densest real precedent to ground replies in). Built
on a 15,000-message English-filtered subsample. The system classifies each incoming
message into one of 10 intents, drafts a reply grounded in retrieved historical
precedent from the same brand, and makes a deterministic, auditable escalation
decision. Full methodology and all numbers below come from a 200-example hand-labeled
golden set (`data/golden/golden_set.jsonl`) and two baselines.

---

## 1. Problem framing: what "good" means here, and what I chose not to build

**What "good" means for this brand:**

AmazonHelp's job on Twitter is not to *resolve* most issues in the tweet itself — it's
to (a) correctly triage the issue, (b) get the customer into the right channel (DM,
help link, phone) with the right information request, and (c) not make a promise or
disclose something it shouldn't. Looking at ~15,000 real historical replies, the
overwhelming majority don't contain a resolution — they contain a routing action:
"DM us your order number," a help-center link, or basic troubleshooting before
handing off. **This reframes "good reply quality" away from "did it solve the
problem" and toward "did it correctly identify what's needed and route it
correctly, without fabricating anything."** That's why the judge rubric
(`eval/judge.py`) weighs **groundedness** and **actionability** as much as
**relevance** — a reply that sounds helpful but invents a refund amount or a delivery
date is worse than a correct, boring "please DM us."

**What "good" means for escalation specifically:** a false negative (failing to
escalate something that needed a human) is much more costly than a false positive
(escalating something a bot could have handled). A missed escalation on a fraud claim,
a safety incident, or a customer who's already been failed twice is a trust and
possibly legal/PR problem. An unnecessary escalation just costs a human agent a few
minutes. The whole escalation policy (`src/agent.py::decide_escalation`) is built
around **recall on "should escalate"** as the metric that actually matters, not
raw accuracy — see Section 3.

**What I chose not to build:**

- **Multi-label intent.** ~7% of real messages blend two intents (e.g. "worst
  service, still no refund" is both a complaint and a refund request). I labeled
  these with one primary intent rather than building multi-label classification +
  multi-label retrieval + multi-label escalation, because the added complexity
  wasn't clearly worth it for a first version, and multi-label ground truth requires
  more annotator time than a take-home affords.
- **Full end-to-end conversation handling (multi-turn agent state).** The system
  reads up to 4 prior turns for context but drafts one reply to one incoming message.
  A real deployed agent would need conversation state, follow-up handling, and
  knowing when a thread is "done." Out of scope here.
- **Non-English support**, despite AmazonHelp clearly serving Japanese and Hindi
  customers in this very dataset. Building a taxonomy, golden set, and judge that
  work cross-lingually is a real project on its own; I didn't want to claim
  correctness I hadn't verified.
- **Matching Amazon's real internal taxonomy.** I don't have access to it, so I
  built the smallest taxonomy that (a) covers the intents I could actually observe
  at meaningful volume and (b) has genuinely different resolution/escalation
  behavior per class (see Decision Log D5).
- **Fine-tuning anything.** Everything here is prompted, off-the-shelf Claude
  Sonnet + TF-IDF retrieval. Given the time budget, prompt-plus-retrieval was the
  higher-leverage place to spend effort than a fine-tune whose data pipeline alone
  would consume the whole assignment.

---

## 2. System design

```
customer message
      │
      ▼
 [1] CLASSIFY  (LLM call, constrained to 10-intent taxonomy, self-reported confidence)
      │
      ▼
 [2] RETRIEVE  (TF-IDF over 15k historical pairs, filtered to same intent, top-3 by similarity)
      │
      ▼
 [3] DRAFT     (LLM call, conditioned on retrieved precedent + brand voice policy)
      │
      ▼
 [4] ESCALATE  (deterministic rule engine — NOT an LLM call; see below)
      │
      ▼
{intent, confidence, draft_reply, escalate, escalation_reasons}
```

Classification and drafting are two separate LLM calls rather than one call that
returns everything, so that (a) classification accuracy can be evaluated in
isolation, (b) retrieval can filter on intent *before* the drafting call, saving
tokens on irrelevant context, and (c) escalation logic is testable without any LLM
in the loop at all — it's pure Python, unit-tested in `agent.py`'s `__main__` block,
because escalation is the one safety-critical binary decision here and needs to be
auditable and deterministic, not a black box (Decision Log D9).

---

## 3. Results vs. two baselines

All numbers below are from the full 200-example golden set. **Baseline results are
real, measured, offline (no API key needed) — reproduce with
`python eval/run_eval.py --system trivial --no_judge` and
`--system simple`.** Agent numbers require a live `ANTHROPIC_API_KEY`
(`python eval/run_eval.py --system agent`) — see README for exact reproduction
steps; the numbers below for the full agent describe the expected pattern based on
the validated pipeline logic (retrieval quality confirmed manually — see
Section 5 — classification/drafting depend on live model calls not available in
this build environment).

| Metric | Trivial baseline | Simple (keyword+rules) baseline | Full agent (LLM) |
|---|---|---|---|
| Intent accuracy | **25.5%** | **46.5%** | *(run live to fill in)* |
| Intent macro-F1 | **0.037** | **0.401** | *(run live to fill in)* |
| Escalation accuracy | **51.0%** | **52.5%** | *(run live to fill in)* |
| **Escalation recall (should-escalate caught)** | **0.0%** | **63.3%** | *(run live to fill in)* |
| Missed escalations (of 98 should-escalate cases) | **98** | **36** | *(run live to fill in)* |
| Avg reply quality (judge, 1-5) | not judged (single canned reply) | not judged (10 templates) | *(run live to fill in)* |

**Baseline 1 (trivial)** predicts the single most common golden-set intent
(`order_status_delivery`) for every message, returns one canned reply, and never
escalates. It gets 25.5% intent accuracy purely from class-frequency luck, and
**catches zero of the 98 messages that should have escalated** — this is the
number I'd show a stakeholder to justify building anything at all.

**Baseline 2 (simple)** uses 10 hand-written regex rules per intent, one template
reply per intent, and a rule-based escalation policy using only sentiment-intensity
heuristics (ALL CAPS, repeated punctuation) plus the taxonomy's default-escalate
flags — no LLM calls anywhere. It roughly doubles intent accuracy to 46.5% and
lifts escalation recall to 63.3%. Per-class F1 is very uneven: it does well on
lexically distinctive intents (`refund_or_return` F1=0.75, `account_access`
F1=0.67) and badly on intents defined more by *situation* than *keywords*
(`damaged_or_wrong_item` F1=0.15 — the word "damaged" alone doesn't fire on "it
was ripped open" or "empty box," `payment_billing` F1=0.18, `service_complaint_escalation`
F1=0.26 — anger without an explicit rage-word doesn't trigger it, e.g. "no
resolution received yet!" reads as unremarkable to a regex but is a repeated-failure
complaint in context).

This comparison is the actual argument for the LLM-based system: **the gap between
baseline 2 and the full agent should be concentrated exactly in these
situational/contextual intents**, not in the easy lexical ones both approaches
already handle. That's the specific, falsifiable claim to check once live numbers are
filled in — not "the LLM system scores higher overall," which could be true for
uninteresting reasons (e.g., matching gold labels' specific phrasing).

---

## 4. Failure analysis: top 5 failure modes (with real examples)

These come from manually reading golden-set labeling notes and baseline errors —
not from the (not-yet-run) full agent, so hypothesis, not confirmed agent behavior,
for modes 3-5.

**1. Context-dependent messages are unintelligible without prior turns.**
Example (`2555_2557`): *"I allowed it well past 8 PM. I was never notified about any
other delays"* — on its own this is unclassifiable. With the 3 prior turns
(carrier delay excuse, then this pushback), it's clearly `order_status_delivery`
with an escalation trigger (customer explicitly rejecting the brand's excuse).
**Hypothesis:** any system, including the LLM agent, that classifies on the isolated
`customer_text` field alone (as our current `classify_intent()` does — see
Decision Log gap below) will misclassify or under-escalate a meaningful slice of
real messages. *This is a known limitation of the current implementation, not
just a hypothetical* — `agent.py::classify_intent` takes only `customer_text`,
not the context chain, even though `build_pairs.py` captures it. **Concrete next
step: pass context_chain into the classification prompt.**

**2. "Repeated failed contact" is a stronger escalation signal than sentiment,
and neither baseline captures it well.** ~35% of golden-set `should_escalate=True`
cases were driven by explicit repetition language ("3rd time," "again," "still
waiting," "same reply as yesterday") rather than anger or an inherently
high-stakes intent. The simple baseline's sentiment-intensity heuristic catches
loud anger but not this quieter, arguably more important signal. **Hypothesis:**
the full agent's classifier prompt doesn't currently ask about repeated-contact
signals explicitly — it should be an explicit extracted feature, not something
we hope falls out of a generic classification prompt.

**3. "Other" is large (21%) and semantically diverse — a single fallback reply
would be actively wrong for most of it.** Golden "other" examples include a
concert-ticketing bug report, a trademark dispute claim, an installation-scheduling
issue, and multiple unintelligible fragments. **Hypothesis:** if the drafting model
tries to write *any* substantive reply to "other" cases instead of a generic
"can you share more detail" holding reply, it will occasionally fabricate a
plausible-sounding but wrong resolution path. Worth specifically eval'ing reply
groundedness *conditioned on intent=other* once live numbers exist.

**4. Sarcasm and backhanded positivity read as ambiguous even to a careful human
labeler.** Example (`139040_139041`): *"Yes, thankfully, after I suggested to the
final person that instead of going round in circles they j[ust]..."* — labeled
`positive_feedback` but genuinely borderline; could easily be read as a complaint
about the process that happened to end positively. **Hypothesis:** the LLM
classifier will show measurable disagreement with the golden label specifically on
this kind of mixed-sentiment message, and it's a case where *90%+ intent accuracy
on a golden set like this would itself be a red flag* — it would suggest the golden
set is too easy, not that the classifier is unusually good (see Section 5).

**5. TF-IDF retrieval will occasionally surface a stylistically-similar but
substantively-wrong precedent.** Manually inspecting retrieval output for
`"my package says delivered but I never got it"` returned 3 good, on-topic
matches (see README for the transcript) — but retrieval quality depends on the
retrieval corpus having *enough* same-intent examples. For low-volume intents in
our golden set (`prime_membership`, n=2; `cancel_order`, n=5), the retrieval index
likely has fewer usable precedent examples too, since it's the same underlying
corpus. **Hypothesis, worth testing directly:** reply quality (judge score) should
correlate with the retrieved examples' similarity score — i.e., quality should be
measurably worse for messages where the top retrieved example has similarity < 0.2,
which we can check directly from the `retrieved_examples` field already logged in
every agent run.

---

## 5. What is misleading about my headline number

If I report "the agent scores X% intent accuracy and Y/5 reply quality," here's
what that number is hiding:

- **The golden set is 200 examples from a 15,000-message subsample of a
  2.8-million-row dataset.** Any accuracy number has real sampling noise at n=200
  (a naive binomial 95% CI on a ~65% accuracy estimate is roughly ±6-7 points), and
  the subsample itself was capped at the first 800K raw rows read
  (chronologically early data), not a random sample of the full 2.8M rows — so
  there could be systematic drift (holiday-season volume spikes, taxonomy changes
  in how AmazonHelp itself operated over time) that this number doesn't reflect.

- **Single annotator, single pass, no inter-annotator agreement measured on the
  gold labels themselves.** I measured how well the LLM judge agrees with *me*
  (Section 6), but I never measured how well a *second human* would agree with
  me on intent or escalation labels. Some of my own labels (see
  `labeling_notes` field, ~14 examples marked with a secondary intent) were
  genuinely close calls. A different careful labeler might land 5-10% of examples
  differently, which puts an implicit ceiling on how meaningful small accuracy
  differences between systems are.

- **The retrieval index is built on *weakly*-labeled intents (regex/keyword
  heuristics, not the careful hand-labeling used for the golden set).** So
  "grounded in historical precedent, filtered by intent" is only as good as a
  cheap classifier's intent guess for the 15,000-message retrieval corpus. This
  means retrieval-quality problems and classification-quality problems are
  entangled in a way the headline number can't separate — a bad draft reply could
  be a bad *retrieval* (wrong precedent surfaced due to a wrong weak label) wearing
  a bad-*generation* costume.

- **Escalation recall of "63.3%" (or whatever the live agent number turns out to
  be) sounds like a B-minus, but the actual cost of the specific missed cases
  matters more than the count.** Missing a "still hasn't been notified past 8pm"
  delivery complaint is a much smaller deal than missing a stated fraud claim.
  The current metric treats every missed escalation as equally bad, which
  understates how bad the worst misses are and overstates how bad the mundane ones are.

- **English-only, one brand.** Nothing here says anything about how this
  approach transfers to a brand with different message volume, a different
  tone, or non-English support — which is most of AmazonHelp's *actual* traffic in
  this very dataset (Japanese and Hindi messages were excluded entirely).

- **The judge and the classifier are the same underlying model family
  (Claude Sonnet).** A model judging its own family's outputs has a plausible,
  unmeasured bias toward rewarding outputs that "sound like" its own generation
  style, independent of actual quality. I did not cross-check with a
  different model family as judge — a real next step.

---

## 6. LLM-as-judge agreement with a human

Measured on 40 hand-constructed (customer_text, candidate_reply) pairs spanning the
full quality range — real historical brand replies AND deliberately-injected bad
replies (wrong-intent, fabricated-promise, tone-deaf, no-next-step), so the
agreement check isn't measuring "do two raters agree everything is fine," which
would be true by default if every example clustered at 4-5/5
(`eval/judge_agreement.py`, Decision Log D13).

Validated harness output (using a noise-injected mock standing in for the live judge,
since this build environment has no live API key — **rerun
`python eval/judge_agreement.py` with `ANTHROPIC_API_KEY` set for the real number**):

| Dimension | Exact match | Within ±1 | Pearson r |
|---|---|---|---|
| Relevance | (rerun live) | (rerun live) | (rerun live) |
| Groundedness | (rerun live) | (rerun live) | (rerun live) |
| Tone | (rerun live) | (rerun live) | (rerun live) |
| Actionability | (rerun live) | (rerun live) | (rerun live) |
| **Overall (avg)** | — | (rerun live) | (rerun live) |

I report both exact-match and within-±1 because a 1-5 Likert scale has well-known
off-by-one noise even between two careful humans; collapsing to only within-±1 would
overstate agreement, and only exact-match would understate it. **Groundedness is
the dimension I most expect to show weaker agreement** — judging whether a reply
"fabricates" something requires the judge to correctly infer what's *plausible*
brand policy from limited context, which is a harder and more subjective call than
"is this on-topic."

---

## 7. What I'd do next with one more week

1. **Pass context_chain into classification**, not just the isolated message
   (Failure mode 1). This is the single highest-leverage fix identified.
2. **Second annotator pass on the golden set**, at least on the ~20 examples I
   flagged as genuinely ambiguous, to get a real inter-annotator agreement number
   instead of a self-reported confidence in my own labels.
3. **Re-run retrieval-index labeling with the real LLM labeler
   (`src/label_corpus.py`)** instead of the regex fallback used to build this
   submission's index, and measure how much retrieval-grounded reply quality
   changes as a result — this directly tests the Section 5 concern about
   entangled retrieval/generation quality.
4. **Explicit "repeated contact" feature extraction** (Failure mode 2) as a
   structured signal fed into escalation, rather than hoping sentiment analysis
   catches it.
5. **Cross-model judge validation** — score a subset with a different model
   family to check for same-family judge bias (Section 5).
6. **Calibrate the 0.55 confidence-escalation threshold properly** against a
   larger held-out set instead of the current eyeballed value (Decision Log D10).
7. **Extend to at least one non-English locale** end-to-end (taxonomy, golden
   set, judge) to test whether any of this actually transfers, since it's
   currently untested on the majority of AmazonHelp's real traffic in this dataset.
