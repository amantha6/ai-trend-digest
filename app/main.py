import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
load_dotenv()

from app.core.rag_chain import build_rag_chain
from app.api.routes import router

chain = None

#yield is the key here, everything before yield runs on startup, everything after runs on shutdown. This is where we build the RAG chain once and reuse it for all requests.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    print("--- Starting AI Trend Digest ---")
    print("--- Building RAG chain ---")
    chain = build_rag_chain()
    print("--- Ready to answer questions! ---")
    yield #server starts here
    print("--- Shutting down ---")

# You run: uvicorn app.main:app
#         ↓
# "🚀 Starting AI Trend Digest..."     ← before yield
# "🔗 Building RAG chain..."           ← before yield
# [HuggingFace model loads ~3 sec]
# [Qdrant connection established]
# "✅ Ready to answer questions!"      ← before yield
#         ↓
#         yield ← SERVER IS LIVE HERE
#         ↓
# [student sends question]  → handled
# [student sends question]  → handled
# [student sends question]  → handled
#         ↓
# You press Ctrl+C
#         ↓
# "👋 Shutting down..."               ← after yield

class ChainMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.chain = chain
        response = await call_next(request)
        return response
# What middleware is: Code that runs on every single request, before it reaches your endpoint. Think of it like a bouncer at a restaurant who checks every customer before they sit down.
app = FastAPI(
    title="AI Trend Digest",
    description="Ask questions about recent AI research trends, grounded in arxiv papers.",
    version="1.0.0",
    lifespan=lifespan,
)

# lifespan=lifespan connects your startup/shutdown function to the app. Without this line, build_rag_chain() never runs and every request fails.
app.add_middleware(ChainMiddleware)
app.include_router(router, prefix="/api/v1")
# This plugs your routes menu into the restaurant. The prefix means every route in routes.py gets /api/v1 prepended:

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)