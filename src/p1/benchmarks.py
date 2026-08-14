import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from genai_assignment.config import Settings, get_settings
from genai_assignment.p1.chunking import chunk_documents
from genai_assignment.p1.embeddings import Embedder
from genai_assignment.p1.eval import load_questions
from genai_assignment.p1.loaders import load_documents
from genai_assignment.p1.models import Chunk
from genai_assignment.p1.qdrant_store import QdrantVectorStore


def run_vector_store_benchmark(
    settings: Settings | None = None,
    top_k: int = 5,
    output_path: Path | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    chunks, vectors, questions, query_vectors = _prepare(settings)
    store = QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection, settings.embedding_dim)

    import faiss

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)

    qdrant_rows = []
    faiss_rows = []
    for question, query_vector in zip(questions, query_vectors, strict=True):
        qdrant_start = time.perf_counter()
        qdrant_hits = store.search(
            query_vector.tolist(),
            top_k=top_k,
            filters=question.get("metadata_filter"),
        )
        qdrant_ms = _elapsed_ms(qdrant_start)
        qdrant_rows.append(
            _row(question, [hit.source_id for hit in qdrant_hits], qdrant_ms)
        )

        faiss_start = time.perf_counter()
        retrieved = _faiss_search_with_filter(
            faiss_index,
            chunks,
            query_vector,
            top_k=top_k,
            filters=question.get("metadata_filter"),
        )
        faiss_ms = _elapsed_ms(faiss_start)
        faiss_rows.append(_row(question, [chunk.source_id for chunk in retrieved], faiss_ms))

    report = {
        "top_k": top_k,
        "qdrant": _summarize(qdrant_rows),
        "faiss": _summarize(faiss_rows),
        "notes": [
            "Both stores use all-MiniLM-L6-v2 normalized dense vectors and inner-product/cosine scoring.",
            "Qdrant applies metadata filters natively; FAISS over-retrieves and post-filters in Python.",
        ],
    }
    output_path = output_path or settings.output_dir / "p1_vector_store_benchmark.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_retrieval_ablation(
    settings: Settings | None = None,
    top_k: int = 5,
    include_reranker: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    chunks, vectors, questions, query_vectors = _prepare(settings)
    tokenized = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    dense_rows = []
    bm25_rows = []
    rrf_rows = []
    reranked_rows = []
    reranker = _load_reranker(settings.reranker_model) if include_reranker else None

    for question, query_vector in zip(questions, query_vectors, strict=True):
        filters = question.get("metadata_filter")

        dense_start = time.perf_counter()
        dense_ranked = _dense_rank(chunks, vectors, query_vector, filters)
        dense_rows.append(_row(question, [chunk.source_id for chunk in dense_ranked[:top_k]], _elapsed_ms(dense_start)))

        bm25_start = time.perf_counter()
        bm25_ranked = _bm25_rank(chunks, bm25, question["question"], filters)
        bm25_rows.append(_row(question, [chunk.source_id for chunk in bm25_ranked[:top_k]], _elapsed_ms(bm25_start)))

        rrf_start = time.perf_counter()
        dense_for_fusion = _dense_rank(chunks, vectors, query_vector, filters)
        bm25_for_fusion = _bm25_rank(chunks, bm25, question["question"], filters)
        fused = _rrf([dense_for_fusion, bm25_for_fusion])
        rrf_rows.append(_row(question, [chunk.source_id for chunk in fused[:top_k]], _elapsed_ms(rrf_start)))

        if reranker is not None:
            rerank_start = time.perf_counter()
            candidates = fused[: min(len(fused), max(top_k * 3, top_k))]
            reranked = _rerank(reranker, question["question"], candidates)
            reranked_rows.append(_row(question, [chunk.source_id for chunk in reranked[:top_k]], _elapsed_ms(rerank_start)))

    report = {
        "top_k": top_k,
        "dense_only": _summarize(dense_rows),
        "bm25_only": _summarize(bm25_rows),
        "dense_bm25_rrf": _summarize(rrf_rows),
        "dense_bm25_rrf_cross_encoder": _summarize(reranked_rows) if reranked_rows else None,
        "notes": [
            "RRF uses k=60 and fuses dense and BM25 rankings after metadata filtering.",
            "The cross-encoder reranks the top fused candidates; first run may include model loading outside measured per-question timings.",
        ],
    }
    output_path = output_path or settings.output_dir / "p1_retrieval_ablation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _prepare(settings: Settings) -> tuple[list[Chunk], np.ndarray, list[dict[str, Any]], np.ndarray]:
    documents = load_documents(settings.corpus_dir)
    chunks = chunk_documents(
        documents,
        settings.default_chunk_size,
        settings.default_chunk_overlap,
        settings.embedding_model,
    )
    embedder = Embedder(settings.embedding_model)
    vectors = embedder.encode([chunk.text for chunk in chunks])
    questions = load_questions(settings.eval_path)
    query_vectors = embedder.encode([item["question"] for item in questions])
    return chunks, vectors, questions, query_vectors


def _faiss_search_with_filter(
    index: Any,
    chunks: list[Chunk],
    query_vector: np.ndarray,
    top_k: int,
    filters: dict[str, str] | None,
) -> list[Chunk]:
    overretrieve = min(len(chunks), max(top_k * 5, top_k))
    _, indices = index.search(np.asarray([query_vector], dtype="float32"), overretrieve)
    results: list[Chunk] = []
    for index_id in indices[0].tolist():
        if index_id < 0:
            continue
        chunk = chunks[index_id]
        if _matches_filter(chunk, filters):
            results.append(chunk)
        if len(results) >= top_k:
            break
    return results


def _dense_rank(
    chunks: list[Chunk],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    filters: dict[str, str] | None,
) -> list[Chunk]:
    scores = vectors @ query_vector
    ranked_indices = np.argsort(scores)[::-1].tolist()
    return [chunks[index] for index in ranked_indices if _matches_filter(chunks[index], filters)]


def _bm25_rank(
    chunks: list[Chunk],
    bm25: BM25Okapi,
    query: str,
    filters: dict[str, str] | None,
) -> list[Chunk]:
    scores = bm25.get_scores(query.lower().split())
    ranked_indices = np.argsort(scores)[::-1].tolist()
    return [chunks[index] for index in ranked_indices if _matches_filter(chunks[index], filters)]


def _rrf(rankings: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    by_id: dict[str, Chunk] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            by_id[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
    return [by_id[chunk_id] for chunk_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _load_reranker(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _rerank(reranker: Any, query: str, chunks: list[Chunk]) -> list[Chunk]:
    if not chunks:
        return []
    scores = reranker.predict([(query, chunk.text) for chunk in chunks])
    ranked = sorted(zip(scores, chunks, strict=True), key=lambda item: float(item[0]), reverse=True)
    return [chunk for _, chunk in ranked]


def _matches_filter(chunk: Chunk, filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    return all(str(chunk.metadata.get(key)) == str(value) for key, value in filters.items())


def _row(question: dict[str, Any], retrieved: list[str], latency_ms: float) -> dict[str, Any]:
    relevant = set(question.get("relevant_doc_ids", []))
    return {
        "id": question["id"],
        "relevant": sorted(relevant),
        "retrieved": retrieved,
        "hit": int(bool(relevant.intersection(retrieved))) if relevant else int(not retrieved),
        "mrr": _mrr(relevant, retrieved),
        "latency_ms": latency_ms,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    answerable = [row for row in rows if row["relevant"]]
    return {
        "questions": len(rows),
        "answerable_questions": len(answerable),
        "hit_rate": round(mean(row["hit"] for row in answerable), 4),
        "mrr": round(mean(row["mrr"] for row in answerable), 4),
        "p50_latency_ms": _percentile([row["latency_ms"] for row in rows], 50),
        "p95_latency_ms": _percentile([row["latency_ms"] for row in rows], 95),
    }


def _mrr(relevant: set[str], retrieved: list[str]) -> float:
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[int(rank)], 3)
    fraction = rank - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
