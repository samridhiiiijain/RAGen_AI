from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("assignment/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    output_dir: Path = Field(default=Path("assignment/outputs"), alias="ASSIGNMENT_OUTPUT_DIR")
    corpus_dir: Path = Field(default=Path("assignment/data/corpus"), alias="ASSIGNMENT_CORPUS_DIR")
    eval_path: Path = Field(default=Path("assignment/data/eval/questions.jsonl"), alias="ASSIGNMENT_EVAL_PATH")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    qdrant_path: Path = Field(default=Path("assignment/outputs/qdrant"), alias="QDRANT_PATH")
    qdrant_collection: str = Field(default="placement_rag_docs", alias="QDRANT_COLLECTION")

    generator_model: str = Field(default="gemini-3.1-flash-lite", alias="GENERATOR_MODEL")
    judge_model: str = Field(default="qwen/qwen3.6-27b", alias="JUDGE_MODEL")
    judge_provider: str = Field(default="groq", alias="JUDGE_PROVIDER")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    default_chunk_size: int = Field(default=450, alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=75, alias="DEFAULT_CHUNK_OVERLAP")
    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings
