````md
# Legal Assistant Chatbot (RAG) — Proof of Concept (Ollama / Gemini)

> ⚠️ **Disclaimer**: this project is a *technical POC* intended to assist with legal research and drafting.  
> It **does not constitute** legal advice, and any output must be reviewed by a qualified lawyer.

## Contents
- **RAG**: chunking + embeddings + vector search (**Chroma**) + BM25 + RRF fusion + answers with **citations**
- **Evaluation**: retrieval metrics (Hit@K, Recall@K, MRR) on a question set
- **Observability**: OpenTelemetry traces (Jaeger) + Prometheus metrics + Grafana dashboard
- **UI**: Streamlit chat interface

## Stack
- Python 3.11+ (tested on macOS)
- FastAPI (backend), Streamlit (frontend)
- ChromaDB (vector store)
- sentence-transformers (embeddings)
- **LLM**:
  - **Ollama** (local, recommended for the POC)
  - **Gemini** (cloud, free with quota)

---

## Quickstart (local)

### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
````

### 2) Build the index (required the first time)

```bash
python scripts/build_index.py
```

### 3) Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

* Swagger (API docs): **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
* Main endpoint: `POST /chat`

> Note: `GET /` may return 404 — this is normal (it’s an API, not a website).

### 4) Run the Streamlit UI

In another terminal:

```bash
streamlit run ui/streamlit_app.py
```

UI: [http://localhost:8501](http://localhost:8501)

---

## Choose your LLM

In `.env`, select the provider:

```env
LLM_PROVIDER=ollama   # or gemini
```

### Option A — Ollama (local)

1. Install the Ollama application
2. Download a model:

```bash
ollama pull llama3.1:8b
ollama list
```

3. `.env` config:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

Direct Ollama test:

```bash
curl http://localhost:11434/api/generate -d '{
  "model":"llama3.1:8b",
  "prompt":"Reply: OK",
  "stream": false
}'
```

### Option B — Gemini (free cloud)

1. Create an API key (Google AI Studio)
2. Install the official SDK:

```bash
pip install google-genai
```

3. `.env` config:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=xxxxxxxxxxxxxxxx
GEMINI_MODEL=gemini-1.5-flash
```

> If you want to verify it’s not Ollama: stop Ollama and retry a `/chat` request.
> If it still responds, it’s Gemini.

---

## Test that the chatbot responds

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain the penalty clause in 2 sentences with sources."}'
```

Expected:

* an `answer` field
* and a `sources/chunks` list (retrieved passages)

---

## Evaluation (retrieval)

⚠️ On macOS / venv, run with `PYTHONPATH=.` so Python can find `app/`:

```bash
PYTHONPATH=. python scripts/evaluate.py --k 5
```

Typical metrics:

* **Hit@K**: at least one relevant doc in the Top K?
* **Recall@K**: share of relevant docs retrieved in the Top K
* **MRR**: does a relevant doc rank high in the results?

---

## Observability (Jaeger + Prometheus + Grafana)

### Option 1 — Start the Docker stack

```bash
docker compose up -d
```

* Jaeger: [http://localhost:16686](http://localhost:16686)
* Prometheus: [http://localhost:9090](http://localhost:9090)
* Grafana: [http://localhost:3000](http://localhost:3000) (admin/admin)

⚠️ If you see an error like:
`jaegertracing/all-in-one:1.61 not found`
edit `docker-compose.yml` to use an existing tag, e.g.:

* `jaegertracing/all-in-one:1.61.0` **or**
* `cr.jaegertracing.io/jaegertracing/all-in-one:1.76.0`

### Option 2 — Disable trace export (if you don’t run Docker)

In `.env`:

```env
OTEL_SDK_DISABLED=true
```

This avoids `Connection refused localhost:4318` errors.

---

## Data

* `data/corpus.jsonl`: small educational corpus (POC)
* Replace it with your own texts (GDPR, contracts, case law, etc.), then rerun:

```bash
python scripts/build_index.py
```

---

## Demo (quick idea)

1. Streamlit: ask 2–3 questions and show **sources**
2. `PYTHONPATH=. python scripts/evaluate.py --k 5`: show Hit@5 / MRR
3. Jaeger: show a trace with `retrieval` then `generation` spans
4. Grafana: show p95 latency + requests/s

