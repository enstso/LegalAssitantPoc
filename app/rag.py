from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from opentelemetry import trace

from .settings import Settings


TRACER = trace.get_tracer(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _tokenize(text: str) -> List[str]:
    # simple, language-agnostic tokenization for BM25
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _rrf_fusion(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion. Returns fused score per item id."""
    scores: Dict[str, float] = {}
    for lst in rank_lists:
        for r, item_id in enumerate(lst):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + r + 1)
    return scores


class LegalRAG:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embedder: Optional[SentenceTransformer] = None

        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_doc_ids: List[str] = []
        self._load_bm25()

        self._reranker = None
        if settings.use_reranker:
            try:
                from sentence_transformers import CrossEncoder

                # small cross encoder; you can swap for a stronger one
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                self._reranker = None

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            # good speed/quality tradeoff for POC
            self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._embedder

    def _embed(self, texts: List[str]) -> np.ndarray:
        model = self._get_embedder()
        emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return emb

    def _bm25_path(self) -> str:
        return os.path.join(self.settings.chroma_dir, "bm25.pkl")

    def _load_bm25(self) -> None:
        path = self._bm25_path()
        if not self.settings.use_bm25:
            return
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.bm25 = obj["bm25"]
            self.bm25_doc_ids = obj["chunk_ids"]
        except Exception:
            self.bm25 = None
            self.bm25_doc_ids = []

    def _vector_search(self, question: str, n: int) -> List[Tuple[str, float]]:
        q_emb = self._embed([question])[0].tolist()
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n,
            include=["distances"],
        )
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        # cosine distance -> similarity proxy
        sims = [1.0 - float(d) for d in distances]
        return list(zip(ids, sims))

    def _bm25_search(self, question: str, n: int) -> List[Tuple[str, float]]:
        if self.bm25 is None:
            return []
        tokens = _tokenize(question)
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:n]
        out: List[Tuple[str, float]] = []
        for i in idx:
            out.append((self.bm25_doc_ids[int(i)], float(scores[int(i)])))
        return out

    def _fetch_chunks_by_id(self, chunk_ids: List[str]) -> Dict[str, Chunk]:
        if not chunk_ids:
            return {}
        got = self.collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        ids = got.get("ids", [])
        docs = got.get("documents", [])
        metas = got.get("metadatas", [])
        out: Dict[str, Chunk] = {}
        for cid, text, meta in zip(ids, docs, metas):
            meta = meta or {}
            out[cid] = Chunk(
                chunk_id=cid,
                doc_id=str(meta.get("doc_id", "")),
                title=str(meta.get("title", "")),
                text=str(text),
                score=0.0,
            )
        return out

    def retrieve(self, question: str) -> List[Chunk]:
        top_k = self.settings.top_k
        cand_n = max(10, top_k * 4)

        with TRACER.start_as_current_span("retrieval") as span:
            span.set_attribute("top_k", top_k)
            vec = self._vector_search(question, cand_n)
            bm = self._bm25_search(question, cand_n) if self.settings.use_bm25 else []

            # RRF uses ranks, not raw scores
            vec_rank = [cid for cid, _ in sorted(vec, key=lambda x: x[1], reverse=True)]
            bm_rank = [cid for cid, _ in sorted(bm, key=lambda x: x[1], reverse=True)]
            fused = _rrf_fusion([vec_rank, bm_rank] if bm_rank else [vec_rank])

            # pick top candidates by fused score
            fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:cand_n]
            cand_ids = [cid for cid, _ in fused_sorted]

            chunks = list(self._fetch_chunks_by_id(cand_ids).values())
            # assign fused score
            fused_map = dict(fused_sorted)
            for c in chunks:
                c.score = float(fused_map.get(c.chunk_id, 0.0))

            # optional rerank using cross-encoder
            if self._reranker is not None and len(chunks) > 1:
                pairs = [(question, c.text) for c in chunks]
                r_scores = self._reranker.predict(pairs).tolist()
                for c, s in zip(chunks, r_scores):
                    c.score = float(s)
                chunks.sort(key=lambda x: x.score, reverse=True)
            else:
                chunks.sort(key=lambda x: x.score, reverse=True)

            final = chunks[:top_k]
            span.set_attribute("returned_chunks", len(final))
            return final

    def _format_context(self, chunks: List[Chunk]) -> str:
        # Keep it compact and citeable.
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[C{i}] {c.title} (doc={c.doc_id}, chunk={c.chunk_id})\n{_normalize_ws(c.text)}")
        return "\n\n".join(parts)

    def generate(self, question: str, chunks: List[Chunk]) -> str:
        provider = self.settings.llm_provider

        with TRACER.start_as_current_span("generation") as span:
            span.set_attribute("provider", provider)
            if provider == "openai":
                return self._generate_openai(question, chunks)
            if provider == "ollama":
                return self._generate_ollama(question, chunks)
            return self._generate_mock(question, chunks)

    def _generate_mock(self, question: str, chunks: List[Chunk]) -> str:
        # Extractive baseline: pick top chunks and assemble an answer with explicit citations.
        if not chunks:
            return "Je ne sais pas répondre avec les sources disponibles."

        # Take 1–3 chunks depending on length
        selected = chunks[: min(3, len(chunks))]
        bullets = []
        for idx, c in enumerate(selected, 1):
            sent = _normalize_ws(c.text)
            # Keep first ~2 sentences
            sents = re.split(r"(?<=[.!?])\s+", sent)
            short = " ".join(sents[:2]).strip()
            bullets.append(f"- {short} [C{idx}]")
        answer = (
            "Réponse (extrait des sources retrouvées) :\n"
            + "\n".join(bullets)
            + "\n\nSources : "
            + ", ".join([f"[C{i}] {c.title}" for i, c in enumerate(selected, 1)])
        )
        return answer

    def _generate_openai(self, question: str, chunks: List[Chunk]) -> str:
        # Uses OpenAI Chat Completions; kept lightweight to avoid framework lock-in.
        try:
            from openai import OpenAI
        except Exception as e:
            return f"OpenAI SDK non installé. Installez `openai` ou utilisez LLM_PROVIDER=mock. Détail: {e}"

        if not self.settings.openai_api_key:
            return "OPENAI_API_KEY manquant. Configurez-le ou utilisez LLM_PROVIDER=mock."

        ctx = self._format_context(chunks)
        system = (
            "Tu es un assistant juridique interne. "
            "Réponds UNIQUEMENT à partir du CONTEXTE fourni. "
            "Si le contexte ne suffit pas, dis 'Je ne sais pas' et demande quelle information manque. "
            "Ajoute des citations de type [C1], [C2] pour chaque affirmation importante. "
            "Ne donne pas de conseil juridique personnalisé."
        )
        user = f"QUESTION: {question}\n\nCONTEXTE:\n{ctx}"

        client = OpenAI(api_key=self.settings.openai_api_key)
        resp = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    def _generate_ollama(self, question: str, chunks: List[Chunk]) -> str:
        ctx = self._format_context(chunks)
        prompt = (
            "Tu es un assistant juridique interne. "
            "Réponds UNIQUEMENT à partir du CONTEXTE. "
            "Si le contexte ne suffit pas, dis 'Je ne sais pas'. "
            "Ajoute des citations [C1], [C2]...\n\n"
            f"QUESTION: {question}\n\nCONTEXTE:\n{ctx}"
        )

        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return str(data.get("response", "")).strip()
        except Exception as e:
            return f"Erreur Ollama ({url}). Détail: {e}"

    def chat(self, question: str) -> Tuple[str, List[Chunk]]:
        chunks = self.retrieve(question)
        answer = self.generate(question, chunks)
        return answer, chunks
