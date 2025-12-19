# Legal Assistant Chatbot (RAG) — Proof of Concept

> ⚠️ **Disclaimer**: This is a *technical* proof‑of‑concept meant to support internal work (research/drafting).
> It is **not** legal advice and must be reviewed by a qualified lawyer.

## What’s included
- **RAG**: chunking + embeddings + vector search (Chroma) + BM25 + RRF fusion + optional rerank + answer with citations
- **Evaluation**: retrieval metrics (Hit@K, MRR, Recall@K) + optional RAGAS generation scoring
- **Observability**: OpenTelemetry traces (Jaeger) + Prometheus metrics + Grafana dashboards
- **UI**: Streamlit chat front‑end

## Tech stack
- Python 3.11+
- FastAPI (backend), Streamlit (UI)
- ChromaDB (vector store)
- sentence-transformers (embeddings)
- OpenTelemetry + Jaeger + Prometheus + Grafana

## Quickstart (local)
### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Build the index
```bash
python scripts/build_index.py
```

### 3) Run the API
```bash
uvicorn app.main:app --reload --port 8000
```
- Healthcheck: http://localhost:8000/health
- Metrics (Prometheus): http://localhost:8000/metrics

### 4) Run the UI
```bash
streamlit run ui/streamlit_app.py
```

## Choose an LLM provider
Set in `.env`:
- `LLM_PROVIDER=mock` (default): no external dependency; produces extractive answers from retrieved context.
- `LLM_PROVIDER=ollama`: uses a local Ollama model.
  - Example: `OLLAMA_MODEL=llama3.1`
- `LLM_PROVIDER=openai`: uses OpenAI API.
  - Provide `OPENAI_API_KEY=...` and `OPENAI_MODEL=gpt-4o-mini` (or similar)

## Evaluation
```bash
python scripts/evaluate.py --k 5
```
Outputs:
- Retrieval: Hit@K, MRR, Recall@K
- (Optional) RAGAS scores if `LLM_PROVIDER=openai` and `ENABLE_RAGAS=true`

## Observability (Docker)
Start the open-source stack (Jaeger + Prometheus + Grafana):
```bash
docker compose up -d
```
- Jaeger UI: http://localhost:16686
- Grafana: http://localhost:3000 (default user/pass: admin/admin)
- Prometheus: http://localhost:9090

## Data
- `data/corpus.jsonl` contains a small, *educational* law-related corpus (written/paraphrased for this POC).
- Replace it with your domain texts (e.g., GDPR, IP case law...) and rerun `scripts/build_index.py`.
