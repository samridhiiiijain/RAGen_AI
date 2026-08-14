from pathlib import Path

from genai_assignment.p1.chunking import chunk_documents
from genai_assignment.p1.eval import _mrr, _ndcg, _recall
from genai_assignment.p1.loaders import load_documents


TEST_TMP = Path("assignment/outputs/unit_tmp")


def test_markdown_loader_extracts_metadata() -> None:
    case_dir = TEST_TMP / "loader"
    case_dir.mkdir(parents=True, exist_ok=True)
    source = case_dir / "sample.md"
    source.write_text(
        "# Sample\n\nThis is searchable text.\n\nMetadata: product=acmeops, doc_type=test\n",
        encoding="utf-8",
    )

    docs = load_documents(case_dir)

    assert len(docs) == 1
    assert docs[0].source_id == "sample"
    assert docs[0].metadata == {"product": "acmeops", "doc_type": "test"}
    assert "Metadata:" not in docs[0].text


def test_chunk_ids_are_stable() -> None:
    case_dir = TEST_TMP / "chunking"
    case_dir.mkdir(parents=True, exist_ok=True)
    source = case_dir / "sample.md"
    source.write_text(
        "one two three four five six seven eight nine ten\nMetadata: product=acmeops",
        encoding="utf-8",
    )
    docs = load_documents(case_dir)

    first = chunk_documents(docs, chunk_size=4, chunk_overlap=1, embedding_model="test-model")
    second = chunk_documents(docs, chunk_size=4, chunk_overlap=1, embedding_model="test-model")

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].metadata["product"] == "acmeops"


def test_retrieval_metric_helpers() -> None:
    relevant = {"a", "c"}
    retrieved = ["b", "a", "d", "c"]

    assert _recall(relevant, retrieved) == 1.0
    assert _mrr(relevant, retrieved) == 0.5
    assert round(_ndcg(relevant, retrieved), 3) > 0.6
