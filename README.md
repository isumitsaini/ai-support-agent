# AI Support Agent for AmazonHelp

An AI support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset for **AmazonHelp**, Amazon's Twitter support account. Given an incoming
customer message, the agent:

1. **Classifies** it into one of 10 intents derived empirically from the data.
2. **Drafts a reply** grounded in how AmazonHelp has actually resolved similar
   issues historically (retrieval-augmented generation over ~15k real prior
   resolutions).
3. **Decides whether to escalate** to a human, with an explicit, auditable reason —
   via a deterministic rule engine, not another LLM call.

See **[REPORT.md](REPORT.md)** for problem framing, baseline comparisons, failure
analysis, and an honest account of what's misleading about the headline numbers.
See **[DECISION_LOG.md](DECISION_LOG.md)** for the 15 non-obvious decisions made
and why.

---

## Quickstart (reproduce headline results in <15 minutes)

```bash
git clone <this-repo>
cd hiver-agent

# 1. Put the Kaggle dataset here (or use the included 15k-row processed subsample —
#    see "Data" section below for exactly which file you need)
cp /path/to/twcs.csv data/raw/twcs.csv

# 2. Set your API key (only needed for the full LLM agent + judge steps;
#    baselines run with no key at all)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run everything
bash scripts/reproduce.sh
```

Without an API key, this reproduces the two **baselines** (real, deterministic,
offline — no cost, <10 seconds) against the 200-example golden set. With a key set,
it additionally runs the full LLM agent and the judge-human agreement check
(~5 minutes total, low API cost — see "Cost" below).

---

## Repo structure

```
data/
  raw/twcs.csv                    # the Kaggle dataset (not committed - see Data section)
  processed/
    amazonhelp_pairs.jsonl        # 15,000 English customer->brand reply pairs, with context chains
    amazonhelp_pairs_labeled.jsonl # same, weakly intent-labeled (for retrieval index)
    retrieval_index.pkl           # TF-IDF index over the above, for grounding replies
  golden/
    golden_sample_unlabeled.jsonl # 200-example stratified sample, pre-labeling
    golden_set.jsonl              # same, with hand-applied gold intent + escalation labels

src/
  taxonomy.py           # the 10-intent taxonomy + resolution/escalation policy per intent
  build_pairs.py         # rebuilds customer->brand conversation pairs from raw twcs.csv
  build_golden_sample.py # stratified sampling for the golden set
  label_corpus.py        # bulk LLM intent-labeling of the retrieval corpus (real path)
  retrieval.py            # TF-IDF retrieval index, filtered by intent
  agent.py                # the full pipeline: classify -> retrieve -> draft -> escalate

eval/
  baselines.py            # trivial + simple (no-LLM) baselines
  judge.py                # LLM-as-judge, 4-dimension rubric
  judge_agreement.py       # measures judge-vs-human agreement on 40 hand-scored examples
  apply_gold_labels.py     # applies hand labels to the golden sample
  golden_set_notes.md      # sampling + labeling methodology
  run_eval.py              # main harness: runs any system over the golden set, computes all metrics
  results/*.json           # output of run_eval.py per system

REPORT.md          # full report (problem framing, results, failure analysis, next steps)
DECISION_LOG.md     # 15 non-obvious decisions and why
scripts/reproduce.sh # one-command reproduction
```

---

## Data

The Kaggle dataset (`twcs.csv`, ~3M rows, ~600MB) is **not committed to this repo**
(too large, and Kaggle's terms prefer you download it directly). Get it from:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Place it at `data/raw/twcs.csv`. **You do not need the full file** — per the
assignment's note that a subsample is expected, `build_pairs.py` reads only the
first N rows (default: 800,000) by default:

```bash
python src/build_pairs.py --brand AmazonHelp --raw data/raw/twcs.csv \
    --out data/processed/amazonhelp_pairs.jsonl --max_rows 800000 --max_pairs 15000
```

The processed 15,000-pair output (`data/processed/amazonhelp_pairs.jsonl`) and the
200-example golden set **are** committed, so you can run the eval harness and
inspect the golden set without downloading the raw Kaggle file at all — you only
need `twcs.csv` if you want to rebuild the pipeline from scratch.

---

## Running each stage individually

```bash
# Rebuild the customer->brand pairs from raw data (only if you have twcs.csv)
python src/build_pairs.py --brand AmazonHelp --raw data/raw/twcs.csv \
    --out data/processed/amazonhelp_pairs.jsonl --max_rows 800000 --max_pairs 15000

# Bulk intent-label the corpus for retrieval (real path, requires API key, ~600 calls)
python src/label_corpus.py --in_path data/processed/amazonhelp_pairs.jsonl \
    --out_path data/processed/amazonhelp_pairs_labeled.jsonl

# Build the retrieval index
python src/retrieval.py --labeled data/processed/amazonhelp_pairs_labeled.jsonl \
    --out data/processed/retrieval_index.pkl

# Rebuild the golden sample (200 examples, stratified, seed=42 - reproducible)
python src/build_golden_sample.py

# Apply hand labels (already done - see eval/apply_gold_labels.py for the labels + rationale)
python eval/apply_gold_labels.py

# Run any system against the golden set
python eval/run_eval.py --system trivial --no_judge   # no API key needed
python eval/run_eval.py --system simple --no_judge    # no API key needed
python eval/run_eval.py --system agent                 # requires API key

# Check judge-human agreement
python eval/judge_agreement.py                          # requires API key

# Try the agent on a single message
python src/agent.py --message "my package says delivered but I never got it"
```

---

## Important note on this specific submission's build environment

This repo was built and tested in a sandboxed environment **without network access
to `api.anthropic.com`** — only package registries (PyPI/npm/GitHub) were reachable.
As a result:

- **The retrieval index (`retrieval_index.pkl`) was built using a fast regex/keyword
  weak-labeler**, not the real LLM-based `label_corpus.py`, because the latter
  requires live API calls. This is clearly marked in `REPORT.md` Section 5 as a
  known limitation of this specific run, and `label_corpus.py` (the intended real
  path) is fully implemented and ready to run with a live key.
- **The full agent (`eval/run_eval.py --system agent`) and the judge-human
  agreement check (`eval/judge_agreement.py`) were not run end-to-end against a
  live model in this submission** — they are fully implemented, unit-tested on
  their non-LLM logic (escalation policy, retrieval ranking, metric computation),
  and validated with mock/synthetic judges to confirm the measurement code is
  correct. **Both baselines and all offline logic (escalation policy, retrieval,
  classification metrics, golden set) were run for real and the numbers in
  REPORT.md for those are genuine, not illustrative.**
- Set `ANTHROPIC_API_KEY` and run `bash scripts/reproduce.sh` to get the real
  full-agent and judge numbers — the harness will produce them in the placeholders
  marked "(rerun live)" in `REPORT.md`.

I'm flagging this explicitly rather than fabricating plausible-looking numbers for
the parts that need a live model, per the assignment's own instruction that "the
proof is worth more than the system" — a fake number would undermine the one thing
this assignment is actually testing.

---

## Cost estimate (for the live-API steps)

Using Claude Sonnet 4.6 pricing at time of writing:
- Classification: ~200 calls x ~400 input tokens x ~100 output tokens ≈ negligible (<$1)
- Drafting: ~200 calls x ~600 input tokens (incl. retrieved examples) x ~150 output tokens ≈ negligible (<$1)
- Judge: ~240 calls (200 golden + 40 agreement-check) x ~300 tokens each ≈ negligible (<$1)
- Full corpus labeling (`label_corpus.py`, optional, only for rebuilding the
  retrieval index with real labels instead of the regex fallback): ~600 batched
  calls (25 messages/call) x ~1500 tokens each ≈ a few dollars.

Total for the golden-set eval path (baselines + agent + judge): well under $5.

---

## Citations / borrowed code

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  (Kaggle, uploader: thoughtvector).
- `langdetect` (PyPI) used for English-language filtering.
- `scikit-learn`'s `TfidfVectorizer` / `cosine_similarity` used for retrieval —
  standard library usage, no borrowed implementation.
- No code was copied from external sources beyond standard library/package usage
  documented above. All prompts, taxonomy design, escalation policy, and eval
  harness logic are original to this submission.
