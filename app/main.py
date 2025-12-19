from __future__ import annotations

import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from opentelemetry import trace

from .observability import (
    REQUESTS_TOTAL,
    REQUEST_LATENCY,
    RETRIEVAL_LATENCY,
    GENERATION_LATENCY,
    RETRIEVED_CHUNKS,
    instrument_fastapi,
    setup_tracing,
    timer,
)
from .rag import LegalRAG
from .schemas import ChatRequest, ChatResponse, RetrievedChunk
from .settings import settings

load_dotenv()

app = FastAPI(title="Legal Assistant POC", version="0.1.0")

# tracing + instrumentation
setup_tracing(settings.otel_service_name, settings.otel_endpoint)
instrument_fastapi(app)

rag: Optional[LegalRAG] = None
TRACER = trace.get_tracer(__name__)


@app.on_event("startup")
def _startup():
    global rag
    rag = LegalRAG(settings)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global rag
    if rag is None:
        raise HTTPException(503, "Index not ready")

    endpoint = "/chat"
    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        with TRACER.start_as_current_span("chat_request") as span:
            start = time.perf_counter()
            try:
                with timer(RETRIEVAL_LATENCY):
                    chunks = rag.retrieve(req.question)
                RETRIEVED_CHUNKS.observe(len(chunks))
                with timer(GENERATION_LATENCY):
                    answer = rag.generate(req.question, chunks)

                # Best-effort trace id
                trace_id = format(span.get_span_context().trace_id, "032x")

                REQUESTS_TOTAL.labels(endpoint=endpoint, status="200").inc()
                return ChatResponse(
                    answer=answer,
                    trace_id=trace_id,
                    chunks=[
                        RetrievedChunk(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id,
                            title=c.title,
                            score=float(c.score),
                        )
                        for c in chunks
                    ],
                )
            except Exception as e:
                REQUESTS_TOTAL.labels(endpoint=endpoint, status="500").inc()
                raise HTTPException(500, f"Erreur interne: {e}") from e
