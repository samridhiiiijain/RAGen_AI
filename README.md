# Gen AI Placement Assignment

This project implements two applied GenAI systems for a cost-efficient RAG question-answering service over a PDF/HTML/Markdown corpus, and an LLM-as-judge evaluation pipeline with explicit bias checks. It includes reproducible ingestion, retrieval evaluation, live Gemini generation, Qwen/Groq as judge for generation responses, cost estimates, benchmark artifacts, and evidence.

## Final Choices

- Problem 1 vector store: Qdrant local persistent mode.
- Problem 1 bonus vector benchmark: FAISS, dense-only comparison against Qdrant.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- Generator model: Gemini 3.1 Flash-Lite.
- Problem 2 judge model: Qwen 3.6 27B via Groq.

## Layout

- `data/corpus/` - synthetic source documents in Markdown, HTML, and generated PDF form.
- `data/eval/questions.jsonl` - gold retrieval and answer evaluation set.
- `src/genai_assignment/` - reusable code for config, ingestion, retrieval, evaluation, and judging.
- `outputs/` - generated indexes, logs, metrics, and reports.

## Environment

Copy `.env.example` to `.env` and fill the keys you want to run:

```powershell
Copy-Item assignment\.env.example assignment\.env
```

Required for generation:

- `GEMINI_API_KEY`

Required for Problem 2 judging:

- `GROQ_API_KEY`

The code is designed so saved generator outputs can be judged later without rerunning Gemini.

## Phase Status

- Phase 1 scaffold: complete.
- Phase 2 Problem 1 runnable RAG/eval: complete.
- Problem 2 implementation: complete with live Gemini generation and live Qwen/Groq judging.

## Problem 1 Commands

Run these from the repository root:

```powershell
$env:PYTHONPATH="assignment\src"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"

python -m genai_assignment.cli p1 index
python -m genai_assignment.cli p1 query "How long does AcmeOps keep encrypted backups?" --filter-product acmeops --filter-doc-type compliance
python -m genai_assignment.cli p1 eval-retrieval --top-k 5
python -m genai_assignment.cli p1 eval-answers --top-k 5
python -m genai_assignment.cli p1 benchmark-stores --top-k 5
python -m genai_assignment.cli p1 ablate-retrieval --top-k 5 --include-reranker
python -m genai_assignment.cli p1 costs
```

If the MiniLM or cross-encoder models are not cached yet, omit the offline
environment variables on the first run.

## Current Problem 1 Results

- Indexed documents: 13 across Markdown, HTML, and PDF.
- Indexed chunks: 13 using chunk size 450 and overlap 75.
- Retrieval eval: HitRate 1.0, Recall 1.0, MRR 0.9412, nDCG 0.9566.
- Live answer eval: EM 0.0, F1 0.6466, heuristic faithfulness 0.8622, answer relevance 0.6466.
- Qdrant benchmark: HitRate 1.0, MRR 0.9412, p95 1.341 ms.
- FAISS benchmark: HitRate 1.0, MRR 0.9412, p95 0.146 ms.
- Reranker ablation: dense+BM25+RRF+cross-encoder improved MRR to 0.9706 with p95 91.391 ms.

Generated artifacts:

- `outputs/p1_retrieval_eval.json`
- `outputs/p1_answer_eval.json`
- `outputs/p1_vector_store_benchmark.json`
- `outputs/p1_retrieval_ablation.json`
- `outputs/p1_cost_estimates.csv`

## Problem 2 Commands

Live run with Gemini 3.1 Flash-Lite generation and Qwen/Groq judging:

```powershell
$env:PYTHONPATH="assignment\src"
$env:GEMINI_API_KEY="..."
$env:GROQ_API_KEY="..."
python -m genai_assignment.cli p2 generate
python -m genai_assignment.cli p2 judge --judge-engine qwen
```

Optional same-family self-enhancement comparison using Gemini as judge over the
same saved generator outputs:

```powershell
python -m genai_assignment.cli p2 judge --judge-engine gemini
```

## Current Problem 2 Live Results

- Suite size: 10 judge cases.
- Compared configs: `baseline` vs `grounded`.
- Live judge: Qwen 3.6 27B via Groq.
- Live generator: Gemini 3.1 Flash-Lite.
- Bias-aware declared winner: no reliable winner.
- Stable aggregation: `baseline` won 2 cases, `grounded` won 0, 6 were ties, and 2 were unstable.
- Raw majority before stability filtering: `baseline` won 4 cases, `grounded` won 0.
- Position flip rate after prompt hardening: 0.2.
- Human validation agreement against `data/p2/human_validation.jsonl`: 0.3.
- Interpretation: the judge pipeline works, but the live judge is not reliable enough to gate releases without human review.
- Confidence intervals: included in `outputs/p2_suite_report.json`.
- Bias probes covered: order flipping, verbosity padding, self-enhancement,
  sycophancy, and score anchors/pairwise mitigation.

Generated artifacts:

- `outputs/p2_generated_outputs.jsonl`
- `outputs/p2_pairwise_verdicts.jsonl`
- `outputs/p2_pairwise_verdicts_qwen_judge.jsonl`
- `outputs/p2_suite_report.json`
- `outputs/p2_suite_report_qwen_judge.json`

## Submission Drafting

- `FINAL_RESULTS.md` summarizes the final live metrics and artifact paths.
