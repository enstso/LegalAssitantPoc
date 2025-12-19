from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_poc")
CORPUS_PATH = os.getenv("CORPUS_PATH", "data/corpus.jsonl")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out


def tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def main() -> None:
    corpus_file = Path(CORPUS_PATH)
    if not corpus_file.exists():
        raise SystemExit(f"Corpus introuvable: {corpus_file.resolve()}")
    docs = []
    for line in corpus_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Reset collection for deterministic rebuild (POC)
    # Note: Chroma doesn't have a portable 'truncate' in all versions; this is a best-effort.
    try:
        existing = collection.get(include=[])
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunk_ids: List[str] = []
    chunk_texts: List[str] = []
    metadatas: List[Dict] = []

    for d in docs:
        doc_id = d["id"]
        title = d.get("title", doc_id)
        text = d.get("text", "")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            cid = f"{doc_id}#c{i}"
            chunk_ids.append(cid)
            chunk_texts.append(ch)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source": d.get("source", ""),
                    "chunk_index": i,
                }
            )

    # Embed + upsert in batches
    BATCH = 64
    for i in range(0, len(chunk_texts), BATCH):
        batch_texts = chunk_texts[i : i + BATCH]
        embs = embedder.encode(batch_texts, normalize_embeddings=True)
        collection.upsert(
            ids=chunk_ids[i : i + BATCH],
            documents=batch_texts,
            metadatas=metadatas[i : i + BATCH],
            embeddings=embs.tolist(),
        )

    # Build BM25 index over the same chunks (for hybrid retrieval)
    tokenized = [tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized)
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(CHROMA_DIR) / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    print(f"✅ Index reconstruit: {len(chunk_ids)} chunks → {CHROMA_DIR}/{COLLECTION_NAME}")
    print("✅ BM25 sauvegardé: bm25.pkl")


if __name__ == "__main__":
    main()
