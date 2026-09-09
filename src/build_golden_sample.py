"""
build_golden_sample.py

Draws a 200-example STRATIFIED random sample from the processed pairs for hand-labeling
into the golden evaluation set.

Sampling method (see eval/golden_set_notes.md for the full writeup - this is the
mandatory "how you sampled" note):
  - Stratify by message length tercile (short/medium/long) so the golden set isn't
    dominated by short, easy-to-classify messages, which are over-represented in
    the raw corpus.
  - Within each stratum, sample uniformly at random (seed=42, reproducible).
  - We deliberately do NOT stratify by keyword-guessed intent, because that would
    bake our own taxonomy assumptions into which examples get selected, biasing
    the eval set toward intents we already anticipated and against edge cases /
    ambiguous "other" cases - exactly the failure mode a good eval set should
    surface, not hide (see report "What is misleading about my headline number").
  - We oversample nothing for sentiment/anger - if angry messages are 8% of the
    corpus, they should be ~8% of the golden set, not artificially inflated, or
    our escalation-rate and reply-quality metrics stop reflecting production
    reality.

Output: data/golden/golden_sample_unlabeled.jsonl (200 rows, to be hand-labeled)
"""
import json
import random

random.seed(42)

IN_PATH = "data/processed/amazonhelp_pairs.jsonl"
OUT_PATH = "data/golden/golden_sample_unlabeled.jsonl"
N_TOTAL = 200


def main():
    records = [json.loads(l) for l in open(IN_PATH)]
    lengths = [(i, len(r["customer_text"])) for i, r in enumerate(records)]
    lengths.sort(key=lambda x: x[1])

    n = len(lengths)
    third = n // 3
    strata = {
        "short": lengths[:third],
        "medium": lengths[third: 2 * third],
        "long": lengths[2 * third:],
    }

    per_stratum = N_TOTAL // 3
    sampled_indices = []
    for name, bucket in strata.items():
        idxs = [i for i, _ in bucket]
        chosen = random.sample(idxs, min(per_stratum, len(idxs)))
        sampled_indices.extend(chosen)

    # top up to exactly N_TOTAL if rounding left us short
    remaining_pool = list(set(range(n)) - set(sampled_indices))
    while len(sampled_indices) < N_TOTAL:
        sampled_indices.append(remaining_pool.pop())

    random.shuffle(sampled_indices)

    with open(OUT_PATH, "w") as f:
        for idx in sampled_indices:
            r = records[idx]
            row = {
                "pair_id": r["pair_id"],
                "customer_text": r["customer_text"],
                "actual_brand_reply": r["brand_reply"],  # kept for reference during labeling, NOT used as ground truth reply
                "context_chain": r.get("context_chain", []),
                # fields to be filled by human labeler:
                "gold_intent": None,
                "gold_should_escalate": None,
                "gold_escalate_reason": None,
                "labeling_notes": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {N_TOTAL} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
