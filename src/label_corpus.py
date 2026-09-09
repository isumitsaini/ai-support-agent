"""
label_corpus.py

Bulk-labels historical customer messages with our intent taxonomy so the
retrieval index (retrieval.py) can filter candidates by intent, not just
lexical similarity.

This is a WEAK LABELING step, not ground truth - errors here degrade retrieval
quality slightly but do not corrupt the golden eval set (which is hand-labeled
separately and independently; see eval/golden_set_notes.md). We accept noisy
labels on the 15k-row retrieval corpus because:
  1. It's used for retrieval (top-3 similarity), which is robust to a
     moderate mislabeling rate,
  2. Hand-labeling 15k rows is out of scope for a take-home,
  3. We validate the *golden* set (what matters for reported metrics) by hand.

Uses batched calls (25 messages/call) to keep cost and time bounded - labeling
15,000 messages this way is ~600 API calls, well under budget for the assignment.
"""
import json
import os
import time
from pathlib import Path

import anthropic

from taxonomy import INTENT_LIST

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 25

SYSTEM = f"""You are labeling customer support messages sent to Amazon's Twitter support account
with an intent from this fixed list ONLY:
{json.dumps(INTENT_LIST)}

Respond with ONLY a JSON array of strings (one intent per input message, same order, same length).
No preamble, no markdown fences, no explanation."""


def label_batch(messages: list) -> list:
    numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user", "content": numbered}],
    )
    text = resp.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        labels = json.loads(text)
    except json.JSONDecodeError:
        # fallback: label everything "other" for this batch rather than crash the run
        labels = ["other"] * len(messages)
    if len(labels) != len(messages):
        labels = (labels + ["other"] * len(messages))[: len(messages)]
    # guard against hallucinated intent names
    labels = [l if l in INTENT_LIST else "other" for l in labels]
    return labels


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/processed/amazonhelp_pairs.jsonl")
    ap.add_argument("--out_path", default="data/processed/amazonhelp_pairs_labeled.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap rows for a quick/cheap run")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.in_path)]
    if args.limit:
        records = records[: args.limit]

    out_f = open(args.out_path, "w")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        texts = [r["customer_text"] for r in batch]
        try:
            labels = label_batch(texts)
        except Exception as e:
            print(f"Batch {i} failed: {e}, retrying once...")
            time.sleep(2)
            labels = label_batch(texts)
        for r, lab in zip(batch, labels):
            r["intent"] = lab
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"Labeled {min(i+BATCH_SIZE, len(records))}/{len(records)}")

    out_f.close()
    print(f"Done -> {args.out_path}")


if __name__ == "__main__":
    main()
