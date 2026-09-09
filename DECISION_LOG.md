# Decision Log

Non-obvious decisions made while building this, and why. Referenced as D1, D2, ... from
code comments so you can trace a design choice back to its rationale.

**D1. Reconstructed threads from flat tweet rows rather than trusting any pre-built
thread field.** `twcs.csv` has no conversation-ID column. We rebuild
customer→brand pairs by following `in_response_to_tweet_id`, and recover up to 4
turns of prior context per pair by walking the chain backwards. Without this, "reply
quality" would be judged against messages stripped of the context that makes them
interpretable (see golden set example `2555_2557`, which is unintelligible without
its 3 prior turns).

**D2. Restricted to direct-reply pairs only, not "any two tweets near each other in
time."** AmazonHelp answers customers in many languages and interleaves conversations
constantly; a time-window heuristic would silently mismatch customers. We only trust
the explicit `in_response_to_tweet_id` link.

**D3. Filtered to English only, using `langdetect`, not a whitelist of ASCII
characters.** AmazonHelp's true handle serves at least English, Japanese, and Hindi
support in this dataset. Mixing languages would corrupt both the TF-IDF retrieval
index and bias the LLM judge (which we did not validate on non-English text).

**D4. Chose AmazonHelp over other brands mainly because of its message volume
(~170k inbound), not brand recognizability.** More volume means the retrieval index
has denser real precedent per intent, which is the part of the system doing the
actual "grounding" work.

**D5. Intent taxonomy has 10 intents + "other," derived from keyword frequency
analysis on the sampled corpus, not from guessing Amazon's real internal category
list (which we don't have access to).** We deliberately stopped adding intents once
two candidate categories would receive identical resolution/escalation treatment -
e.g. "billing question" and "unexpected charge" merged into one `payment_billing`
class, because splitting them wouldn't change downstream agent behavior.

**D6. "other" is a first-class intent with `default_escalate=True`, not a residual
bucket that gets a generic reply.** 21% of our golden set falls in "other" - garbled
fragments, off-topic posts, and edge cases outside the 9 real categories (concert
ticketing bugs, trademark disputes, installation scheduling). Forcing these into the
9 "real" categories to hit a higher intent-accuracy number would be actively
dangerous: it means confidently drafting a resolution for a case the system doesn't
actually understand.

**D7. TF-IDF retrieval, not embeddings.** Chosen for zero-GPU, zero-vector-DB setup
cost, and because these are short, template-heavy support replies where lexical
overlap correlates strongly with the right precedent. We did NOT rigorously
A/B this against embeddings for this submission - see report "what I'd do next" -
this is a documented tradeoff, not a proven optimum.

**D8. Retrieval filters candidates by (weakly-labeled) intent BEFORE ranking by
similarity, rather than ranking by similarity alone.** A lexically similar
wrong-intent example is worse than a lexically-distant right-intent one, because the
drafting model anchors heavily on the *type* of resolution shown, not just word
overlap.

**D9. Escalation is a separate, deterministic rule-based policy layer, not another
LLM call.** Escalation is the one safety-critical binary decision in this system. We
want it auditable (you can read `decide_escalation()` and know exactly why any
message escalated), deterministic (same input always gives the same decision, so it
can be unit-tested without an API key - see `eval/test escalation` block in
`agent.py`'s `__main__`), and separable from generation quality in evaluation (a bad
draft reply and a bad escalation decision are different failure modes and we don't
want one LLM call's mistake to silently cause the other).

**D10. Classification confidence below 0.55 forces escalation, regardless of which
intent was predicted.** This threshold was picked by inspecting the golden set's
"other" and ambiguous cases, not tuned against the full eval set (that would be
circular). It's explicitly flagged as arbitrary and worth calibrating properly with
more data - see report "what's misleading."

**D11. Golden-set examples with two defensible intents (~7% of the set) are labeled
with a single PRIMARY intent plus a note, not multi-labeled.** The rest of the
pipeline (retrieval filtering, escalation policy) is single-label. Multi-labeling the
golden set but not the pipeline would make the eval numbers incomparable to what the
system actually does.

**D12. The LLM judge scores candidate replies against a rubric, not against the
brand's own historical reply as "ground truth."** Real historical AmazonHelp replies
are themselves inconsistent quality - some are curt, some slightly wrong. Treating
them as gold would cap our system's judged quality at the historical average and
penalize genuinely better replies.

**D13. Judge-human agreement was measured on a 40-example set deliberately
constructed to span bad-to-good replies (via injected wrong/tone-deaf/fabricated
replies), not just historical replies (which cluster near "fine").** If every
example in the agreement check scores 4-5, any two raters will "agree" by
default, and the measured agreement number is meaningless. See `eval/judge_agreement.py`.

**D14. Report's "misleading headline number" section is written before, not after,
seeing final metrics** in the sense that we identified the traps (small golden set,
single annotator, no non-English coverage, subsample not full corpus, retrieval
index uses weak labels not the same golden-quality labels) as an inherent property of
the *method*, not post-hoc excuses discovered after unfavorable numbers came in.

**D15. Chose JSON Lines over Parquet for all intermediate data.** The sandboxed
environment used to build this didn't have `pyarrow` preinstalled; JSONL keeps the
reproduction path free of an extra native-dependency install step, trading a bit of
file size and load speed for one less thing that can break in someone else's
environment during the 15-minute reproduction window.
