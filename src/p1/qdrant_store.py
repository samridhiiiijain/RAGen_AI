from pathlib import Path
from typing import Any

from genai_assignment.p1.models import Chunk, RetrievedChunk


class QdrantVectorStore:
    def __init__(self, path: Path, collection: str, vector_size: int) -> None:
        from qdrant_client import QdrantClient

        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection = collection
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(item.name == self.collection for item in collections):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=self._vector_params(self.vector_size),
        )

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            self._point_struct(
                id=chunk.chunk_id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source_id": chunk.source_id,
                    "source_path": chunk.source_path,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def delete_sources(self, source_ids: list[str]) -> None:
        if not source_ids:
            return
        for source_id in source_ids:
            point_ids = self._point_ids_for_source(source_id)
            if point_ids:
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=self._point_ids_list(point_ids),
                )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        qdrant_filter = _to_qdrant_filter(filters)
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        results: list[RetrievedChunk] = []
        for hit in hits:
            payload: dict[str, Any] = hit.payload or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(hit.id),
                    source_id=str(payload.get("source_id", "")),
                    source_path=str(payload.get("source_path", "")),
                    text=str(payload.get("text", "")),
                    score=float(hit.score),
                    metadata=payload,
                )
            )
        return results

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def _point_ids_for_source(self, source_id: str) -> list[str]:
        found, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=_to_qdrant_filter({"source_id": source_id}),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        return [str(point.id) for point in found]

    @staticmethod
    def _vector_params(vector_size: int) -> Any:
        from qdrant_client.http.models import Distance, VectorParams

        return VectorParams(size=vector_size, distance=Distance.COSINE)

    @staticmethod
    def _point_struct(**kwargs: Any) -> Any:
        from qdrant_client.http.models import PointStruct

        return PointStruct(**kwargs)

    @staticmethod
    def _point_ids_list(point_ids: list[str]) -> Any:
        from qdrant_client.http.models import PointIdsList

        return PointIdsList(points=point_ids)


def _to_qdrant_filter(filters: dict[str, str] | None) -> Any:
    if not filters:
        return None
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
    )
