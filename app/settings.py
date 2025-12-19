from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # Retrieval
    top_k: int = _env_int("TOP_K", 5)
    use_bm25: bool = _env_bool("USE_BM25", True)
    use_reranker: bool = _env_bool("USE_RERANKER", False)

    # Storage
    chroma_dir: str = os.getenv("CHROMA_DIR", ".chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "legal_poc")

    # Observability
    otel_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "legal-assistant-poc")

    # Evaluation
    enable_ragas: bool = _env_bool("ENABLE_RAGAS", False)


settings = Settings()
