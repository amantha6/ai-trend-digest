# AI Trend Digest

> A RAG-powered API that helps AI students ask questions about recent research trends and get plain-English answers grounded in real arxiv papers — with citations.

---

## What it is

A REST API that fetches 250 recent AI research papers from arxiv, converts them into searchable vectors stored in Qdrant, and uses Groq's Llama 3 to generate student-friendly answers with paper citations.

---

## How it was built (one-time setup)

```
Project setup → Fetch papers → Embed + store → RAG chain → FastAPI server → Docker + Render
    venv           250 papers      FastEmbed       retriever      /query           live URL
    API keys       arxiv API       384-dim          + Groq LLM     /health          public API
    folders        5 topics        Qdrant cloud     wired up        + logging        auto-deploy
```

---

## What happens on every request

```
Student asks question
        ↓
FastAPI receives POST /query
        ↓
FastEmbed converts question → 384-dim vector
        ↓
Qdrant finds top 5 most similar paper chunks (cosine similarity)
        ↓
Groq Llama 3 generates plain-English answer from chunks
        ↓
Answer + paper citations returned as JSON
        ↓
Query logged to logs/queries.json
```

---

## Tools & why each was used

| Tool | Why |
|---|---|
| arxiv API | Free source of 250 recent AI papers across 5 topics |
| FastEmbed | Converts text → 384-dim vectors locally, free, lightweight (~50MB) |
| Qdrant | Cloud vector database — stores and searches embeddings by cosine similarity |
| LangChain | Wires retriever + LLM into one RetrievalQA chain |
| Groq + Llama 3 | Free LLM API — generates student-friendly answers from retrieved chunks |
| FastAPI | REST API with auto-generated /docs, validation, and query logging |
| Docker | Packages the entire app so it runs identically anywhere |
| Render | Free cloud host — deploys automatically on every git push |

---

## Project structure

```
ai-trend-digest/
├── app/
│   ├── core/
│   │   ├── ingest.py        ← fetches arxiv papers & stores embeddings
│   │   └── rag_chain.py     ← retrieval + generation logic
│   ├── api/
│   │   └── routes.py        ← API endpoints
│   └── main.py              ← FastAPI app entry point
├── data/                    ← cached paper data (papers.json)
├── logs/                    ← query logs for eval (queries.json)
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── .env                     ← API keys (never commit this)
```

---

## Live details

| | |
|---|---|
| API | https://ai-trend-digest.onrender.com/api/v1/query |
| Pretty API | https://ai-trend-digest.onrender.com/api/v1/query/pretty |
| Docs | https://ai-trend-digest.onrender.com/docs |
| Papers | 250 papers · 5 topics · LLMs, diffusion, RL, CV transformers, AI agents |
| Vectors | ~1062 chunks · 384 dimensions · cosine similarity |

---

## Setup & run locally

```bash
# 1. Clone the repo
git clone https://github.com/amantha6/ai-trend-digest
cd ai-trend-digest

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip3 install -r requirements.txt

# 4. Add your API keys to .env
GROQ_API_KEY=your-groq-key
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-key

# 5. Run ingestion (one time only)
python3 -m app.core.ingest

# 6. Start the server
python3 -m uvicorn app.main:app --reload --port 8000
```

---

## Example request

```bash
curl -s -X POST http://localhost:8000/api/v1/query/pretty \
  -H "Content-Type: application/json" \
  -d '{"question": "What is new in AI agents?"}'
```

```
============================================================
QUESTION: What is new in AI agents?
============================================================

ANSWER:
Recent research in AI agents focuses on improving autonomy,
interpretability, and real-world deployment...

SOURCES:
  1. Redefining AI Red Teaming in the Agentic Era
     Published: 2026-05-05
     URL: http://arxiv.org/abs/2605.04019v1

  2. Agentic-imodels: Evolving agentic interpretability tools
     Published: 2026-05-05
     URL: http://arxiv.org/abs/2605.03808v1
============================================================
```

---

## Run with Docker

```bash
docker build -t ai-trend-digest .

docker run -p 8000:8000 \
  -e GROQ_API_KEY=your-key \
  -e QDRANT_URL=your-url \
  -e QDRANT_API_KEY=your-key \
  ai-trend-digest
```

---

## What's next

- [ ] Add RAGAS evaluation scores
- [ ] Build a simple web UI for students
- [ ] Add monitoring and drift detection
- [ ] Auto-refresh papers weekly via scheduled ingestion