"""
build_pairs.py

Reconstructs (customer_message -> brand_reply) pairs for a single brand from the
raw Twitter Customer Support dataset (twcs.csv), with light cleaning.

Why this exists (see decision log D1-D3):
- twcs.csv is a flat table of tweets, not threads. We rebuild threads by following
  in_response_to_tweet_id / response_tweet_id links.
- We only keep pairs where a brand's outbound tweet is a *direct* reply to an
  inbound (customer) tweet, and we recover up to N turns of prior context so
  a reply can be evaluated in context, not in isolation.
- We restrict to English (langdetect) because the brand's replies are templated
  per-locale and mixing languages would confuse both retrieval and the judge.

Usage:
    python src/build_pairs.py --brand AmazonHelp --raw data/raw/twcs.csv \
        --out data/processed/amazonhelp_pairs.parquet --max_rows 400000
"""
import argparse
import re
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
CASE_REF_RE = re.compile(r"\^[A-Za-z]{1,3}\b")  # agent sign-off codes like ^TN, ^RR


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = URL_RE.sub("[link]", t)
    t = t.strip()
    return t


def strip_mentions_for_display(t: str) -> str:
    return MENTION_RE.sub("", t).strip()


def is_english(t: str) -> bool:
    t = MENTION_RE.sub("", t)
    t = URL_RE.sub("", t)
    if len(t.strip()) < 3:
        return False
    try:
        return detect(t) == "en"
    except LangDetectException:
        return False


def get_context_chain(by_id: pd.DataFrame, start_id: int, max_turns: int = 4) -> list:
    """Walk backwards via in_response_to_tweet_id to recover prior turns."""
    chain = []
    cur = start_id
    seen = set()
    for _ in range(max_turns):
        if cur not in by_id.index or cur in seen:
            break
        seen.add(cur)
        row = by_id.loc[cur]
        chain.append({
            "tweet_id": int(cur),
            "author_id": row["author_id"],
            "inbound": bool(row["inbound"]),
            "text": row["text"],
            "created_at": row["created_at"],
        })
        parent = row.get("in_response_to_tweet_id")
        if pd.isna(parent):
            break
        cur = int(parent)
    return list(reversed(chain))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--raw", default="data/raw/twcs.csv")
    ap.add_argument("--out", default="data/processed/pairs.jsonl")
    ap.add_argument("--max_rows", type=int, default=None, help="cap raw rows read (for speed on subsample)")
    ap.add_argument("--max_pairs", type=int, default=40000, help="cap final pairs kept")
    args = ap.parse_args()

    print(f"Loading raw data from {args.raw} ...")
    df = pd.read_csv(args.raw, nrows=args.max_rows)
    df["tweet_id"] = df["tweet_id"].astype(int)
    by_id = df.set_index("tweet_id", drop=False)

    brand_replies = df[
        (df["author_id"] == args.brand)
        & (df["inbound"] == False)
        & (df["in_response_to_tweet_id"].notna())
    ].copy()
    brand_replies["in_response_to_tweet_id"] = brand_replies["in_response_to_tweet_id"].astype(int)
    print(f"Brand outbound tweets: {len(brand_replies)}")

    brand_replies = brand_replies[brand_replies["in_response_to_tweet_id"].isin(by_id.index)]
    brand_replies["customer_tweet_id"] = brand_replies["in_response_to_tweet_id"]
    brand_replies["customer_is_inbound"] = brand_replies["customer_tweet_id"].map(by_id["inbound"])
    brand_replies = brand_replies[brand_replies["customer_is_inbound"] == True]
    print(f"Direct customer->brand pairs: {len(brand_replies)}")

    # Clean + language filter (sampled progressively to control cost)
    records = []
    for _, row in brand_replies.iterrows():
        cust_text_raw = by_id.loc[row["customer_tweet_id"], "text"]
        cust_text = clean_text(cust_text_raw)
        brand_text = clean_text(row["text"])
        if not cust_text or not brand_text:
            continue
        if not is_english(cust_text):
            continue
        context = get_context_chain(by_id, int(row["customer_tweet_id"]), max_turns=4)
        records.append({
            "pair_id": f"{row['customer_tweet_id']}_{row['tweet_id']}",
            "customer_tweet_id": int(row["customer_tweet_id"]),
            "brand_tweet_id": int(row["tweet_id"]),
            "customer_text": strip_mentions_for_display(cust_text),
            "brand_reply": strip_mentions_for_display(brand_text),
            "created_at": row["created_at"],
            "context_chain": context,
        })
        if len(records) >= args.max_pairs:
            break

    out_df = pd.DataFrame(records)
    print(f"Final English-filtered pairs kept: {len(out_df)}")
    out_df.to_json(args.out, orient="records", lines=True, force_ascii=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
