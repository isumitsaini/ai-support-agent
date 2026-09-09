"""
run_eval.py

The main evaluation harness. Runs the full agent (and both baselines) over the
200-example golden set and computes:
  - Intent classification accuracy + per-class precision/recall/F1
  - Escalation decision accuracy, precision, recall (recall on "should escalate"
    matters most - a missed escalation is worse than an unnecessary one)
  - Reply quality via LLM-as-judge (relevance/groundedness/tone/actionability)
  - Cost and latency per message (for the "what's misleading" report section)

Outputs a JSON results file per system to eval/results/ and prints a summary table.

Usage:
    python eval/run_eval.py --system agent --index data/processed/retrieval_index.pkl
    python eval/run_eval.py --system trivial
    python eval/run_eval.py --system simple
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, "src")

from taxonomy import INTENT_LIST


def load_golden(path="data/golden/golden_set.jsonl"):
    return [json.loads(l) for l in open(path)]


def compute_classification_metrics(golden, predictions):
    """predictions: list of predicted intent strings, aligned with golden order."""
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    correct = 0
    for gold_row, pred_intent in zip(golden, predictions):
        gold_intent = gold_row["gold_intent"]
        if pred_intent == gold_intent:
            correct += 1
            per_class[gold_intent]["tp"] += 1
        else:
            per_class[gold_intent]["fn"] += 1
            per_class[pred_intent]["fp"] += 1

    accuracy = correct / len(golden)
    per_class_metrics = {}
    for cls in INTENT_LIST:
        tp, fp, fn = per_class[cls]["tp"], per_class[cls]["fp"], per_class[cls]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    macro_f1 = sum(m["f1"] for m in per_class_metrics.values()) / len(per_class_metrics)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class_metrics}


def compute_escalation_metrics(golden, predictions):
    """predictions: list of bool, aligned with golden order."""
    tp = fp = tn = fn = 0
    for gold_row, pred_escalate in zip(golden, predictions):
        gold_escalate = gold_row["gold_should_escalate"]
        if gold_escalate and pred_escalate:
            tp += 1
        elif gold_escalate and not pred_escalate:
            fn += 1  # MISSED escalation - the costly error
        elif not gold_escalate and pred_escalate:
            fp += 1  # unnecessary escalation - cheap error
        else:
            tn += 1

    accuracy = (tp + tn) / len(golden)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall on "should escalate" - THE key safety metric
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall_on_should_escalate": recall,
        "missed_escalations": fn,
        "unnecessary_escalations": fp,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def run_system(system_name: str, golden: list, retriever=None, judge_replies: bool = True):
    from agent import classify_intent, draft_reply, decide_escalation
    from baselines import trivial_baseline, simple_baseline

    predictions = []
    latencies = []

    for row in golden:
        text = row["customer_text"]
        t0 = time.time()

        if system_name == "trivial":
            pred = trivial_baseline(text)
        elif system_name == "simple":
            pred = simple_baseline(text)
        elif system_name == "agent":
            cls = classify_intent(text)
            examples = retriever.query(text, intent=cls["intent"], k=3) if retriever else []
            reply = draft_reply(text, cls["intent"], examples)
            esc = decide_escalation(text, cls["intent"], cls["confidence"])
            pred = {
                "intent": cls["intent"],
                "confidence": cls["confidence"],
                "draft_reply": reply,
                "escalate": esc.escalate,
                "escalation_reasons": esc.reasons,
            }
        else:
            raise ValueError(system_name)

        latencies.append(time.time() - t0)
        predictions.append(pred)

    intents = [p["intent"] for p in predictions]
    escalations = [p["escalate"] for p in predictions]
    replies = [p["draft_reply"] for p in predictions]

    cls_metrics = compute_classification_metrics(golden, intents)
    esc_metrics = compute_escalation_metrics(golden, escalations)

    quality_scores = []
    if judge_replies:
        from judge import judge_reply
        for row, reply, intent in zip(golden, replies, intents):
            score = judge_reply(row["customer_text"], reply, intent)
            quality_scores.append(score)

    result = {
        "system": system_name,
        "n_examples": len(golden),
        "classification": cls_metrics,
        "escalation": esc_metrics,
        "avg_latency_sec": sum(latencies) / len(latencies) if latencies else None,
        "quality_scores": quality_scores,
        "predictions": predictions,
    }

    if quality_scores:
        for dim in ["relevance", "groundedness", "tone", "actionability", "overall"]:
            vals = [s[dim] for s in quality_scores if s.get(dim) is not None]
            result[f"avg_{dim}"] = sum(vals) / len(vals) if vals else None

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["trivial", "simple", "agent"])
    ap.add_argument("--index", default="data/processed/retrieval_index.pkl")
    ap.add_argument("--golden", default="data/golden/golden_set.jsonl")
    ap.add_argument("--no_judge", action="store_true", help="skip LLM-judge step (classification/escalation only, no API cost)")
    ap.add_argument("--limit", type=int, default=None, help="run on first N golden examples only (cheap smoke test)")
    args = ap.parse_args()

    golden = load_golden(args.golden)
    if args.limit:
        golden = golden[: args.limit]

    retriever = None
    if args.system == "agent":
        from retrieval import ResolutionRetriever
        retriever = ResolutionRetriever.load(args.index)

    result = run_system(args.system, golden, retriever=retriever, judge_replies=not args.no_judge)

    Path("eval/results").mkdir(parents=True, exist_ok=True)
    out_path = f"eval/results/{args.system}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== {args.system.upper()} ===")
    print(f"Intent accuracy:     {result['classification']['accuracy']:.3f}")
    print(f"Intent macro-F1:     {result['classification']['macro_f1']:.3f}")
    print(f"Escalation accuracy: {result['escalation']['accuracy']:.3f}")
    print(f"Escalation recall (should-escalate caught): {result['escalation']['recall_on_should_escalate']:.3f}")
    print(f"Missed escalations:  {result['escalation']['missed_escalations']} / {result['n_examples']}")
    if not args.no_judge:
        print(f"Avg reply quality (overall): {result.get('avg_overall'):.2f} / 5")
    print(f"Saved full results -> {out_path}")


if __name__ == "__main__":
    main()
