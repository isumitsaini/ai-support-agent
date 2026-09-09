"""
retrieval.py

Builds a lightweight TF-IDF retrieval index over historical (customer_text -> brand_reply)
pairs, so the reply generator can be *grounded* in real precedent rather than
free-generating a plausible-sounding but possibly policy-violating reply.

Why TF-IDF and not embeddings (decision log D7):
- This needs to run in <15 min on a laptop with no GPU and no vector DB dependency.
  TF-IDF + cosine gets ~80% of the retrieval quality of embeddings for this task
  (short, template-heavy support replies) at near-zero setup cost. We measure this
  tradeoff directly in eval/retrieval_quality.py rather than asserting it.
- We additionally filter candidates to the SAME intent class before ranking by
  similarity, which matters more than embedding quality here: a lexically similar
  but wrong-intent example is worse than a lexically distant but right-intent one.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResolutionRetriever:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.records = []  # list of dicts: customer_text, brand_reply, intent

    def fit(self, records: list):
        """records: list of dicts with at least 'customer_text', 'brand_reply', 'intent'."""
        self.records = records
        texts = [r["customer_text"] for r in records]
        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, text: str, intent: str = None, k: int = 3):
        if self.vectorizer is None:
            raise RuntimeError("Call fit() first")
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.matrix)[0]

        if intent is not None:
            mask = np.array([r.get("intent") == intent for r in self.records])
            if mask.sum() >= k:
                sims = np.where(mask, sims, -1.0)

        top_idx = np.argsort(sims)[::-1][:k]
        results = []
        for i in top_idx:
            if sims[i] <= 0:
                continue
            r = self.records[i]
            results.append({
                "customer_text": r["customer_text"],
                "brand_reply": r["brand_reply"],
                "intent": r.get("intent"),
                "similarity": float(sims[i]),
            })
        return results

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "records": self.records,
            }, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.matrix = data["matrix"]
        obj.records = data["records"]
        return obj


def build_index_from_labeled(labeled_jsonl_path: str, out_path: str):
    """Build retrieval index from intent-labeled historical pairs."""
    records = [json.loads(l) for l in open(labeled_jsonl_path)]
    retriever = ResolutionRetriever()
    retriever.fit(records)
    retriever.save(out_path)
    print(f"Indexed {len(records)} records -> {out_path}")
    return retriever


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/processed/amazonhelp_pairs_labeled.jsonl")
    ap.add_argument("--out", default="data/processed/retrieval_index.pkl")
    args = ap.parse_args()
    build_index_from_labeled(args.labeled, args.out)
