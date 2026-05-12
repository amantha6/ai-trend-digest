import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class SourceModel(BaseModel):
    title: str
    url: str
    published: str
    topic: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    timestamp: str


router = APIRouter()


@router.post("/query", response_model=AnswerResponse)
async def query_papers(request: Request, body: QuestionRequest):
    """Returns a structured JSON response."""
    chain = request.state.chain

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        from app.core.rag_chain import query
        result = query(body.question, chain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": body.question,
        "answer": result["answer"],
        "sources": result["sources"],
    }
    with open("logs/queries.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return AnswerResponse(
        question=body.question,
        answer=result["answer"],
        sources=[SourceModel(**s) for s in result["sources"]],
        timestamp=datetime.now().isoformat(),
    )


@router.post("/query/pretty", response_class=PlainTextResponse)
async def query_papers_pretty(request: Request, body: QuestionRequest):
    """Returns a human-readable plain text response."""
    chain = request.state.chain

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        from app.core.rag_chain import query
        result = query(body.question, chain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": body.question,
        "answer": result["answer"],
        "sources": result["sources"],
    }
    with open("logs/queries.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    divider = "=" * 60
    sources_text = ""
    for i, s in enumerate(result["sources"], 1):
        sources_text += f"\n  {i}. {s['title']}"
        sources_text += f"\n     Published: {s['published']}"
        sources_text += f"\n     Topic: {s['topic']}"
        sources_text += f"\n     URL: {s['url']}\n"

    return f"""
{divider}
QUESTION: {body.question}
{divider}

ANSWER:
{result["answer"]}

SOURCES:
{sources_text}
Timestamp: {datetime.now().isoformat()}
{divider}
"""


@router.get("/health")
async def health_check():
    """Health check endpoint — used by Render to verify the server is running."""
    return {
        "status": "healthy",
        "service": "AI Trend Digest",
        "version": "1.0.0"
    }