# Golden Evaluation Set — Sampling & Labeling Notes

**File:** `data/golden/golden_set.jsonl` (200 examples)

## Sampling method

Stratified random sample by customer-message length (short/medium/long terciles),
~67 examples per stratum, seed=42, drawn from the 15,000-message processed pool
(`src/build_golden_sample.py`). Reproducible: rerunning the script with the same
seed reproduces the same 200 `pair_id`s.

We deliberately did **not** stratify by keyword-guessed intent. Stratifying by our
own taxonomy guesses would bias the golden set toward intents we already anticipated
and away from the edge cases and ambiguous "other" bucket — exactly the failure mode
a good eval set should surface, not hide. We also did not oversample for anger/
sentiment: if angry messages are ~8% of the corpus, they're ~8% of the golden set,
not artificially inflated, so escalation-rate and reply-quality metrics reflect
realistic production traffic rather than a stress-test-only distribution.

## Labeling method

One labeler (me), single pass, applying `src/taxonomy.py` definitions as the
labeling guide. For every example, I read up to 4 prior conversation turns
(`context_chain` field) before labeling — many messages are unintelligible in
isolation (see Report Failure Mode 1).

**Escalation label decision procedure** (applied uniformly, not case-by-case):
1. Legal threat, fraud claim, safety/health issue, or credential/PII request
   present → escalate=True, regardless of intent.
2. Else, if the context chain shows this is the customer's 2nd+ contact about the
   same unresolved issue (explicit repetition language: "again," "3rd time,"
   "still waiting," "same reply as yesterday") → escalate=True, regardless of
   intent's default flag.
3. Else, escalate = that intent's `default_escalate` flag from `taxonomy.py`.

## Known limitations (see also Report Section 5)

- **Single annotator, no inter-annotator agreement measured on gold labels.**
  ~14 of 200 examples (7%) had a genuinely defensible secondary intent, recorded
  in `labeling_notes` but not used as an official multi-label.
- **2 examples (1%) reference external content not visible to the labeler**
  ("just have a look at the messages I have received today [link]") — labeled
  `other` / escalate=True on the theory that insufficient-context cases should
  fail safe to human review, not force a guess.
- Final intent distribution is uneven by design (it reflects real traffic, not a
  balanced eval set): `order_status_delivery` 25.5%, `other` 21.0%,
  `service_complaint_escalation` 15.0%, down to `prime_membership` at 1.0% (n=2) —
  metrics on the smallest classes (prime_membership, cancel_order) should be read
  as directional, not statistically reliable, given n<10.

## Reproduction

```bash
python src/build_golden_sample.py        # regenerates the unlabeled 200-example sample
python eval/apply_gold_labels.py          # applies the hand labels, prints distribution summary
```
