import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from genai_assignment.config import Settings, get_settings
from genai_assignment.p1.eval import load_questions
from genai_assignment.p1.models import QueryRequest
from genai_assignment.p1.service import RagService


def evaluate_answers(
    use_gold_fixtures: bool = False,
    top_k: int = 5,
    settings: Settings | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    questions = load_questions(settings.eval_path)
    service = None if use_gold_fixtures else RagService(settings)
    output_path = output_path or settings.output_dir / "p1_answer_eval.json"
    partial_path = settings.output_dir / "p1_answer_eval_rows.jsonl"
    existing_rows = {} if use_gold_fixtures else _load_existing_rows(partial_path)

    rows: list[dict[str, Any]] = []
    for item in questions:
        if not use_gold_fixtures and item["id"] in existing_rows:
            rows.append(existing_rows[item["id"]])
            continue
        if use_gold_fixtures:
            answer = item["expected_answer"]
            contexts = []
            no_context = not item.get("relevant_doc_ids")
            usage = {"mode": "gold_fixture"}
            timings = {"total": 0.0}
        else:
            response = service.query(  # type: ignore[union-attr]
                QueryRequest(
                    question=item["question"],
                    top_k=top_k,
                    filters=item.get("metadata_filter"),
                )
            )
            answer = response.answer
            contexts = [context.model_dump() for context in response.contexts]
            no_context = response.no_context
            usage = response.usage
            timings = response.timings_ms

        expected = item["expected_answer"]
        row = {
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "expected_answer": expected,
            "exact_match": _exact_match(answer, expected),
            "f1": _token_f1(answer, expected),
            "faithfulness": _heuristic_faithfulness(answer, contexts, no_context, use_gold_fixtures),
            "answer_relevance": _heuristic_relevance(answer, expected),
            "no_context": no_context,
            "contexts": contexts,
            "usage": usage,
            "timings_ms": timings,
        }
        rows.append(row)
        if not use_gold_fixtures:
            _append_row(partial_path, row)

    summary = {
        "mode": "gold_fixture" if use_gold_fixtures else "live_generation",
        "top_k": top_k,
        "questions": len(rows),
        "exact_match": round(mean(row["exact_match"] for row in rows), 4),
        "f1": round(mean(row["f1"] for row in rows), 4),
        "faithfulness": round(mean(row["faithfulness"] for row in rows), 4),
        "answer_relevance": round(mean(row["answer_relevance"] for row in rows), 4),
        "notes": [
            "EM/F1 are deterministic lexical metrics against expected_answer.",
            "Fixture mode uses gold answers and should be replaced by live_generation before final submission if API keys are available.",
            "Faithfulness/relevance are heuristic in this offline report; live LLM-judge scoring can be added over the saved rows.",
        ],
        "rows": rows,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _exact_match(answer: str, expected: str) -> int:
    return int(_normalize(answer) == _normalize(expected))


def _token_f1(answer: str, expected: str) -> float:
    answer_tokens = _tokens(answer)
    expected_tokens = _tokens(expected)
    if not answer_tokens and not expected_tokens:
        return 1.0
    if not answer_tokens or not expected_tokens:
        return 0.0
    common = set(answer_tokens).intersection(expected_tokens)
    overlap = sum(min(answer_tokens.count(token), expected_tokens.count(token)) for token in common)
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(expected_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def _heuristic_faithfulness(
    answer: str,
    contexts: list[dict[str, Any]],
    no_context: bool,
    use_gold_fixtures: bool,
) -> float:
    if use_gold_fixtures:
        return 1.0
    if no_context:
        return 1.0 if "don't know" in answer.lower() or "do not know" in answer.lower() else 0.0
    context_text = " ".join(str(context.get("text", "")) for context in contexts).lower()
    answer_terms = [token for token in _tokens(answer) if len(token) > 3]
    if not answer_terms:
        return 0.0
    supported = sum(1 for token in answer_terms if token in context_text)
    return round(supported / len(answer_terms), 4)


def _heuristic_relevance(answer: str, expected: str) -> float:
    return _token_f1(answer, expected)


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
