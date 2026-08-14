from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    source_id: str
    path: Path
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)
    checksum: str


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    source_path: str
    text: str
    metadata: dict[str, Any]
    chunk_index: int
    start_word: int
    end_word: int


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    source_path: str
    text: str
    score: float
    metadata: dict[str, Any]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filters: dict[str, str] | None = None
    min_score: float = 0.2


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    contexts: list[RetrievedChunk]
    no_context: bool
    timings_ms: dict[str, float]
    usage: dict[str, int | float | str | None] = Field(default_factory=dict)
