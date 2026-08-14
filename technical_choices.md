# Technical Choices

## Problem 1: Vector Store

Primary choice: Qdrant local persistent mode.

Why: Qdrant gives a production-shaped API, metadata filtering, local persistence,
and simple Docker/cloud migration. It is stronger for this assignment than a
pure in-process FAISS setup because filtering and payload management are first
class, which the requirements explicitly ask for.

Tradeoffs: Qdrant has more moving parts than FAISS or Chroma and is slightly
heavier for a tiny corpus. For very small demos, FAISS can be faster to start.

Bonus comparison: FAISS dense-only benchmark. This shows we understand the
speed/simplicity tradeoff without turning the project into two full RAG services.

## Embeddings

Choice: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions.

Why: It is lightweight, free to run locally, and good enough for short
product-policy documents.

Tradeoffs: It is weaker than larger modern embedding models on hard semantic
matching and multilingual queries. The assignment values cost and reproducibility,
so the local model is a good fit.

## Problem 2: Generator and Judge

Generator: Gemini 3.1 Flash-Lite.

Why: Gemini 3.1 Flash-Lite is a cost-efficient Gemini model, remains in the
Google/Gemini family, and is strong enough for the assignment's generation
tasks.

Judge: Qwen 3.6 27B through Groq (`qwen/qwen3.6-27b`).

Why: It is a different model family from Gemini, which directly supports the
assignment's self-enhancement bias discussion. The judge is used with a strict
rubric, JSON schema, order flipping, and validation against human labels.

Tradeoffs: Qwen may disagree with a human evaluator or cluster scores narrowly,
so we include adversarial probes, confidence intervals, and a validation sample.

## Bonus Scope

Problem 1 bonuses:

- Qdrant vs FAISS retrieval benchmark.
- Dense vs dense+BM25/RRF vs reranked retrieval ablation.
- Reproducible 100K/1M/10M cost calculator.

Problem 2 bonuses:

- Self-enhancement probe comparing Gemini-as-judge baseline with Qwen judge.
- Offline replay/cache so judge experiments can rerun without generation calls.
- Confidence intervals for winner, win rate, flip rate, and validation agreement.
