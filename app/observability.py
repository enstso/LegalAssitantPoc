from __future__ import annotations

import time
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from prometheus_client import Counter, Histogram


REQUESTS_TOTAL = Counter(
    "legal_assistant_requests_total",
    "Total requests to the Legal Assistant API",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "legal_assistant_request_latency_seconds",
    "Latency (seconds) per endpoint",
    ["endpoint"],
)

RETRIEVAL_LATENCY = Histogram(
    "legal_assistant_retrieval_latency_seconds",
    "Latency (seconds) for retrieval step",
)

GENERATION_LATENCY = Histogram(
    "legal_assistant_generation_latency_seconds",
    "Latency (seconds) for generation step",
)

RETRIEVED_CHUNKS = Histogram(
    "legal_assistant_retrieved_chunks",
    "Number of chunks returned by retrieval",
    buckets=(1, 2, 3, 5, 8, 13, 21),
)


def setup_tracing(service_name: str, otlp_endpoint: str) -> None:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Instrument common libraries
    RequestsInstrumentor().instrument()


def instrument_fastapi(app) -> None:
    FastAPIInstrumentor.instrument_app(app)


class timer:
    """Simple context manager to observe durations."""

    def __init__(self, hist: Histogram):
        self.hist = hist
        self.start: Optional[float] = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.start is None:
            return
        duration = time.perf_counter() - self.start
        self.hist.observe(duration)
