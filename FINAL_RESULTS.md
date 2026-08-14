# Final Results

## Chosen Models and Store

- Problem 1 vector store: Qdrant local persistent mode.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions.
- Generator: Gemini 3.1 Flash-Lite.
- Problem 2 judge: Qwen 3.6 27B via Groq, `qwen/qwen3.6-27b`.

## Problem 1 Results

Corpus:

- 13 documents across Markdown, HTML, and PDF.
- 13 indexed chunks with chunk size 450 and overlap 75.
- Metadata filters tested with product/doc-type fields.

Retrieval eval, `k=5`:

| Metric | Value |
|---|---:|
| HitRate | 1.0 |
| Recall | 1.0 |
| MRR | 0.9412 |
| nDCG | 0.9566 |
| Context precision | 0.5324 |
| Retrieval p50 | 9.538 ms |
| Retrieval p95 | 21.734 ms |

Live answer eval:

| Metric | Value |
|---|---:|
| Exact match | 0.0 |
| Token F1 | 0.6466 |
| Heuristic faithfulness | 0.8622 |
| Answer relevance | 0.6466 |

Bonus benchmark:

| Store | HitRate | MRR | p95 latency |
|---|---:|---:|---:|
| Qdrant | 1.0 | 0.9412 | 1.341 ms |
| FAISS | 1.0 | 0.9412 | 0.146 ms |

Retrieval ablation:

| Config | HitRate | MRR | p95 latency |
|---|---:|---:|---:|
| Dense only | 1.0 | 0.9412 | 0.084 ms |
| BM25 only | 1.0 | 0.8451 | 0.314 ms |
| Dense + BM25 RRF | 1.0 | 0.9265 | 0.263 ms |
| Dense + BM25 RRF + cross-encoder | 1.0 | 0.9706 | 91.391 ms |

## Problem 2 Results

Suite:

- 10 pairwise judge cases.
- Config A: `baseline`.
- Config B: `grounded`.
- Each pair judged in both AB and BA order.

Live hardened Qwen judge result:

| Metric | Value |
|---|---:|
| Raw baseline wins | 4 |
| Raw grounded wins | 0 |
| Stable baseline wins | 2 |
| Stable grounded wins | 0 |
| Ties | 6 |
| Unstable cases | 2 |
| Stable cases | 8 |
| Position flip rate | 0.2 |
| Manual validation agreement | 0.3 |

Declared winner: no reliable winner.

Reason: two cases changed preference across AB/BA order, and manual-label agreement remained low. The pipeline flags this for human review instead of trusting raw judge scores.

## Key Artifacts

- `outputs/p1_retrieval_eval.json`
- `outputs/p1_answer_eval.json`
- `outputs/p1_vector_store_benchmark.json`
- `outputs/p1_retrieval_ablation.json`
- `outputs/p1_cost_estimates.csv`
- `outputs/p2_generated_outputs.jsonl`
- `outputs/p2_pairwise_verdicts_qwen_judge.jsonl`
- `outputs/p2_suite_report_qwen_judge.json`


