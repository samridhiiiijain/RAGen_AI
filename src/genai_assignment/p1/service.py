from genai_assignment.config import Settings, get_settings
from genai_assignment.logging_utils import Timer, append_jsonl
from genai_assignment.p1.embeddings import Embedder
from genai_assignment.p1.generator import GeminiGenerator, fallback_no_context_answer
from genai_assignment.p1.models import QueryRequest, QueryResponse
from genai_assignment.p1.qdrant_store import QdrantVectorStore


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = Embedder(self.settings.embedding_model)
        self.store = QdrantVectorStore(
            self.settings.qdrant_path,
            self.settings.qdrant_collection,
            self.settings.embedding_dim,
        )
        self.generator = (
            GeminiGenerator(self.settings.gemini_api_key, self.settings.generator_model)
            if self.settings.gemini_api_key
            else None
        )

    def query(self, request: QueryRequest) -> QueryResponse:
        total_timer = Timer()
        embed_timer = Timer()
        vector = self.embedder.encode([request.question])[0].tolist()
        embed_ms = embed_timer.elapsed_ms()

        retrieval_timer = Timer()
        contexts = self.store.search(vector, request.top_k, request.filters)
        retrieval_ms = retrieval_timer.elapsed_ms()

        grounded_contexts = [item for item in contexts if item.score >= request.min_score]
        no_context = not grounded_contexts
        usage: dict[str, int | float | str | None] = {}

        generation_timer = Timer()
        if no_context:
            answer = fallback_no_context_answer()
        elif self.generator:
            answer, usage = self.generator.answer(request.question, grounded_contexts)
        else:
            answer = fallback_no_context_answer()
            usage = {"warning": "GEMINI_API_KEY is not set; generation was skipped."}
        generation_ms = generation_timer.elapsed_ms()

        response = QueryResponse(
            answer=answer,
            citations=sorted({item.source_id for item in grounded_contexts}),
            contexts=grounded_contexts,
            no_context=no_context,
            timings_ms={
                "embedding": embed_ms,
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": total_timer.elapsed_ms(),
            },
            usage=usage,
        )

        append_jsonl(
            self.settings.output_dir / "query_logs.jsonl",
            {
                "question": request.question,
                "top_k": request.top_k,
                "filters": request.filters,
                "contexts_returned": len(grounded_contexts),
                "top_score": grounded_contexts[0].score if grounded_contexts else None,
                "timings_ms": response.timings_ms,
                "usage": usage,
            },
        )
        return response
