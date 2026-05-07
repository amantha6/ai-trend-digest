import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

#APIrouteer is a mini app that holds all the endpoints instead of defining all routs in one giant file
#HTTP exception is used to raise errors when something goes wrong, it will return the right error code and status message
#BaseModel is the base class for defining the data shapes, when a student sends a request to your API, pydantic automatically validates that the data has the right fields and types before the code even runs

#3block 2: request and response models
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

#why define these models, two reasons
#1. validation, if a student sends a request that doesn't match the expected format, pydantic will automatically return a 422 error with details about what went wrong
#2. documentation, when you define these models, FastAPI can automatically generate API docs that show the expected request and response formats, making it easier for students to understand how to interact with your
#API. FastAPI reads these models and auto-generates interactive API docs at /docs. Anyone can open that page and test your API without writing any code. That's something you can show in interviews.

#Block 3: API router and endpoints
router = APIRouter()

@router.post("/query", response_model=AnswerResponse)
async def query_papers(request: Request, body: QuestionRequest):
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
#@router.post("/query") — this decorator registers the function as a POST endpoint at the /query URL. POST because we're sending data (the question) in the request body, not the URL.
# response_model=AnswerResponse — tells FastAPI to validate the return value matches our AnswerResponse shape and strip any extra fields. Automatic output validation for free.
# async def — makes the endpoint asynchronous. FastAPI can handle other requests while this one waits for Groq to respond, instead of blocking everything.

# Why wrap in try/except? If Groq times out or Qdrant hiccups, we don't want the whole server to crash. We catch the error and return a clean 500 response with a helpful message instead.
# Why log every query? Remember on Saturday's plan we said to log queries for eval. Every question a student asks gets saved here. On Sunday you'll use these logs to run RAGAS evaluation and get your resume metrics. This is also what separates a portfolio project from a toy — real observability.
# SourceModel(**s) — unpacks each source dictionary into a SourceModel object. The ** operator spreads the dict keys as keyword arguments. So {"title": "...", "url": "..."} becomes SourceModel(title="...", url="...").

#block 4: Health Check Endpoint
@router.get("/health")
async def health_check():
    """Simple health check — used by Render to verify the server is running."""
    return {
        "status": "healthy",
        "service": "AI Trend Digest",
        "version": "1.0.0"
    }
# Why a health check? When you deploy to Render later, it pings this endpoint every 30 seconds to make sure your server is alive. If it returns anything other than 200, Render restarts the server automatically. It's also the first thing you call to verify a fresh deployment worked.