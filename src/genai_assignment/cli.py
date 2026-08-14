import json
from pathlib import Path

import typer
from rich import print

from genai_assignment.config import get_settings
from genai_assignment.p1.answer_eval import evaluate_answers
from genai_assignment.p1.benchmarks import run_retrieval_ablation, run_vector_store_benchmark
from genai_assignment.p1.costs import estimate_costs
from genai_assignment.p1.eval import evaluate_retrieval
from genai_assignment.p1.ingest import ingest
from genai_assignment.p1.models import QueryRequest
from genai_assignment.p1.service import RagService
from genai_assignment.p2.pipeline import generate_outputs, judge_suite


app = typer.Typer(help="Gen AI placement assignment utilities.")
p1 = typer.Typer(help="Problem 1 RAG commands.")
p2 = typer.Typer(help="Problem 2 LLM-as-judge commands.")
app.add_typer(p1, name="p1")
app.add_typer(p2, name="p2")


@p1.command()
def index(
    corpus_dir: Path | None = typer.Option(None),
    chunk_size: int | None = typer.Option(None),
    chunk_overlap: int | None = typer.Option(None),
) -> None:
    result = ingest(corpus_dir=corpus_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(result)


@p1.command()
def query(
    question: str,
    top_k: int = 5,
    filter_product: str | None = None,
    filter_doc_type: str | None = None,
) -> None:
    filters = {}
    if filter_product:
        filters["product"] = filter_product
    if filter_doc_type:
        filters["doc_type"] = filter_doc_type
    response = RagService().query(QueryRequest(question=question, top_k=top_k, filters=filters or None))
    print(json.dumps(response.model_dump(), indent=2))


@p1.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("genai_assignment.p1.app:app", host=host, port=port, reload=False)


@p1.command()
def eval_retrieval(top_k: int = 5) -> None:
    summary = evaluate_retrieval(top_k=top_k)
    printable = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


@p1.command()
def eval_answers(top_k: int = 5, use_gold_fixtures: bool = False) -> None:
    summary = evaluate_answers(top_k=top_k, use_gold_fixtures=use_gold_fixtures)
    printable = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


@p1.command()
def costs() -> None:
    rows = estimate_costs()
    for row in rows:
        print(row)


@p1.command()
def benchmark_stores(top_k: int = 5) -> None:
    report = run_vector_store_benchmark(top_k=top_k)
    print(json.dumps(report, indent=2))


@p1.command()
def ablate_retrieval(top_k: int = 5, include_reranker: bool = True) -> None:
    report = run_retrieval_ablation(top_k=top_k, include_reranker=include_reranker)
    print(json.dumps(report, indent=2))


@app.command()
def show_config() -> None:
    settings = get_settings()
    print(settings.model_dump())


@p2.command()
def generate(use_fixtures: bool = False) -> None:
    outputs = generate_outputs(use_fixtures=use_fixtures)
    print({"generated_outputs": len(outputs), "use_fixtures": use_fixtures})


@p2.command()
def judge(use_fixtures: bool = False, judge_engine: str = "qwen") -> None:
    report = judge_suite(use_fixtures=use_fixtures, judge_engine=judge_engine)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    app()
