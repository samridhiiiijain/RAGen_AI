import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from genai_assignment.config import Settings, get_settings
from genai_assignment.logging_utils import Timer
from genai_assignment.p1.embeddings import Embedder
from genai_assignment.p1.qdrant_store import QdrantVectorStore


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def evaluate_retrieval(
    settings: Settings | None = None,
    top_k: int = 5,
    output_path: Path | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    embedder = Embedder(settings.embedding_model)
    store = QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection, settings.embedding_dim)
    questions = load_questions(settings.eval_path)

    rows: list[dict[str, Any]] = []
    for item in questions:
        timer = Timer()
        vector = embedder.encode([item["question"]])[0].tolist()
        hits = store.search(vector, top_k=top_k, filters=item.get("metadata_filter"))
        elapsed_ms = timer.elapsed_ms()

        relevant = set(item.get("relevant_doc_ids", []))
        retrieved = [hit.source_id for hit in hits]
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "relevant_doc_ids": sorted(relevant),
                "retrieved_doc_ids": retrieved,
                "hit": _hit(relevant, retrieved),
                "recall": _recall(relevant, retrieved),
                "mrr": _mrr(relevant, retrieved),
                "ndcg": _ndcg(relevant, retrieved),
                "context_precision": _context_precision(relevant, retrieved),
                "latency_ms": elapsed_ms,
                "top_score": hits[0].score if hits else None,
            }
        )

    answerable_rows = [row for row in rows if row["relevant_doc_ids"]]
    summary = {
        "top_k": top_k,
        "questions": len(rows),
        "answerable_questions": len(answerable_rows),
        "hit_rate": round(mean(row["hit"] for row in answerable_rows), 4),
        "recall": round(mean(row["recall"] for row in answerable_rows), 4),
        "mrr": round(mean(row["mrr"] for row in answerable_rows), 4),
        "ndcg": round(mean(row["ndcg"] for row in answerable_rows), 4),
        "context_precision": round(mean(row["context_precision"] for row in answerable_rows), 4),
        "p50_latency_ms": _percentile([row["latency_ms"] for row in rows], 50),
        "p95_latency_ms": _percentile([row["latency_ms"] for row in rows], 95),
        "rows": rows,
    }
    output_path = output_path or settings.output_dir / "p1_retrieval_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _hit(relevant: set[str], retrieved: list[str]) -> int:
    if not relevant:
        return 1 if not retrieved else 0
    return int(any(doc_id in relevant for doc_id in retrieved))


def _recall(relevant: set[str], retrieved: list[str]) -> float:
    if not relevant:
        return 1.0 if not retrieved else 0.0
    return len(relevant.intersection(retrieved)) / len(relevant)


def _mrr(relevant: set[str], retrieved: list[str]) -> float:
    if not relevant:
        return 0.0
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def _ndcg(relevant: set[str], retrieved: list[str]) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), len(retrieved))
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def _context_precision(relevant: set[str], retrieved: list[str]) -> float:
    if not retrieved:
        return 1.0 if not relevant else 0.0
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved)) / len(retrieved)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[int(rank)], 3)
    fraction = rank - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)
