#!/usr/bin/env bash
# Reproduces the headline results end-to-end. Should complete in <15 minutes
# on a laptop with an ANTHROPIC_API_KEY set, using the included 15k-row subsample
# (the full 2.8M-row twcs.csv is NOT required - see README for how to point at it
# instead if you want to rebuild the subsample yourself).
set -e

echo "== 1. Installing dependencies =="
pip install -r requirements.txt --break-system-packages -q

echo "== 2. Baselines (no API key needed, <10 sec) =="
python eval/run_eval.py --system trivial --no_judge
python eval/run_eval.py --system simple --no_judge

echo "== 3. Building retrieval index (weak-labeled, no API key needed, ~30 sec) =="
python src/retrieval.py --labeled data/processed/amazonhelp_pairs_labeled.jsonl --out data/processed/retrieval_index.pkl

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "!! ANTHROPIC_API_KEY not set - skipping the full LLM agent eval and judge steps."
  echo "!! Baselines above are real, offline, reproducible numbers."
  echo "!! Set ANTHROPIC_API_KEY and re-run to get full-agent + judge results:"
  echo "!!   python eval/run_eval.py --system agent"
  echo "!!   python eval/judge_agreement.py"
  exit 0
fi

echo "== 4. Full LLM agent on golden set (requires API key, ~3-5 min for 200 examples) =="
python eval/run_eval.py --system agent

echo "== 5. Judge-human agreement check (requires API key, ~1 min for 40 examples) =="
python eval/judge_agreement.py

echo ""
echo "Done. See eval/results/*.json for full metrics and REPORT.md for analysis."
