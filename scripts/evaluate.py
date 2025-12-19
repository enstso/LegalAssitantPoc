from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from app.rag import LegalRAG
from app.settings import Settings

load_dotenv()


@dataclass
class RetrievalMetrics:
    hit_at_k: float
    recall_at_k: float
    mrr: float


def load_eval(path: str) -> List[Dict]:
    p = Path(path)
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def compute_retrieval_metrics(rag: LegalRAG, rows: List[Dict], k: int) -> RetrievalMetrics:
    hits = 0
    recalls = []
    rr = []

    for r in rows:
        q = r["question"]
        rel_doc_ids = set(r["relevant_ids"])
        chunks = rag.retrieve(q)
        # keep only top-k (rag settings may already be k, but be safe)
        chunks = chunks[:k]
        retrieved_doc_ids = [c.doc_id for c in chunks]

        # Hit@K
        hit = any(doc_id in rel_doc_ids for doc_id in retrieved_doc_ids)
        hits += 1 if hit else 0

        # Recall@K
        found = len(rel_doc_ids.intersection(retrieved_doc_ids))
        recalls.append(found / max(1, len(rel_doc_ids)))

        # MRR
        rank = None
        for i, doc_id in enumerate(retrieved_doc_ids, 1):
            if doc_id in rel_doc_ids:
                rank = i
                break
        rr.append(0.0 if rank is None else 1.0 / rank)

    return RetrievalMetrics(
        hit_at_k=hits / max(1, len(rows)),
        recall_at_k=sum(recalls) / max(1, len(recalls)),
        mrr=sum(rr) / max(1, len(rr)),
    )


def generation_proxy_metrics(answer: str, relevant_doc_ids: List[str]) -> Dict[str, float]:
    # Very simple proxy: do we cite at least one relevant doc id or a context marker [C1]?
    cites_context = 1.0 if "[C" in answer else 0.0
    cites_relevant = 1.0 if any(doc in answer for doc in relevant_doc_ids) else 0.0
    return {
        "has_context_citations": cites_context,
        "mentions_relevant_docid": cites_relevant,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", default="eval/questions.jsonl")
    parser.add_argument("--k", type=int, default=int(os.getenv("TOP_K", "5")))
    args = parser.parse_args()

    # Override top_k for evaluation
    s = Settings()
    object.__setattr__(s, "top_k", args.k)  # type: ignore[misc]

    rag = LegalRAG(s)
    rows = load_eval(args.eval_path)

    rm = compute_retrieval_metrics(rag, rows, k=args.k)
    print("=== Retrieval ===")
    print(f"Hit@{args.k}:   {rm.hit_at_k:.3f}")
    print(f"Recall@{args.k}: {rm.recall_at_k:.3f}")
    print(f"MRR:        {rm.mrr:.3f}")

    # Lightweight end-to-end check (proxy)
    print("\n=== Generation (proxy) ===")
    acc = {"has_context_citations": 0.0, "mentions_relevant_docid": 0.0}
    for r in rows:
        ans, chunks = rag.chat(r["question"])
        prox = generation_proxy_metrics(ans, r["relevant_ids"])
        for k2 in acc:
            acc[k2] += prox[k2]
    for k2 in acc:
        acc[k2] /= max(1, len(rows))
        print(f"{k2}: {acc[k2]:.3f}")

    print("\nTip: pour des métriques LLM-as-a-judge (faithfulness, answer relevancy...), installez requirements-eval.txt et activez ENABLE_RAGAS=true.")


if __name__ == "__main__":
    main()
