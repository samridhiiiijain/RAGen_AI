import hashlib
import uuid

from genai_assignment.p1.models import Chunk, SourceDocument


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        words = doc.text.split()
        step = chunk_size - chunk_overlap
        for chunk_index, start in enumerate(range(0, len(words), step)):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if not chunk_text:
                continue
            stable = "|".join(
                [
                    doc.source_id,
                    str(chunk_index),
                    doc.checksum,
                    str(chunk_size),
                    str(chunk_overlap),
                    embedding_model,
                ]
            )
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable))
            metadata = {
                **doc.metadata,
                "source_id": doc.source_id,
                "source_path": str(doc.path),
                "chunk_index": chunk_index,
                "start_word": start,
                "end_word": end,
                "source_checksum": doc.checksum,
                "chunk_checksum": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embedding_model,
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=doc.source_id,
                    source_path=str(doc.path),
                    text=chunk_text,
                    metadata=metadata,
                    chunk_index=chunk_index,
                    start_word=start,
                    end_word=end,
                )
            )
    return chunks
