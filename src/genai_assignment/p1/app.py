from fastapi import FastAPI

from genai_assignment.p1.models import QueryRequest, QueryResponse
from genai_assignment.p1.service import RagService


app = FastAPI(title="Gen AI Assignment RAG Service")
service = RagService()


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "indexed_chunks": service.store.count()}


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return service.query(request)
