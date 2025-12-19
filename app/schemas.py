from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    score: float
    text: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    chunks: List[RetrievedChunk]
    trace_id: Optional[str] = None
