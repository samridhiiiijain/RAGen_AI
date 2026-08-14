from dataclasses import dataclass
from pathlib import Path

from genai_assignment.config import Settings, get_settings
from genai_assignment.p1.chunking import chunk_documents
from genai_assignment.p1.embeddings import Embedder
from genai_assignment.p1.loaders import load_documents
from genai_assignment.p1.qdrant_store import QdrantVectorStore


@dataclass(frozen=True)
class IngestResult:
    documents: int
    chunks: int
    collection_count: int
    chunk_size: int
    chunk_overlap: int


def ingest(
    corpus_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    settings = settings or get_settings()
    corpus_dir = corpus_dir or settings.corpus_dir
    chunk_size = chunk_size or settings.default_chunk_size
    chunk_overlap = settings.default_chunk_overlap if chunk_overlap is None else chunk_overlap

    documents = load_documents(corpus_dir)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap, settings.embedding_model)

    embedder = Embedder(settings.embedding_model)
    vectors = embedder.encode([chunk.text for chunk in chunks]).tolist()

    store = QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection, settings.embedding_dim)
    store.delete_sources([doc.source_id for doc in documents])
    store.upsert_chunks(chunks, vectors)

    return IngestResult(
        documents=len(documents),
        chunks=len(chunks),
        collection_count=store.count(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
