#block 1 Import necessary libraries
import os
import json
import arxiv
from dotenv import load_dotenv
load_dotenv()  # loads .env FIRST before anything else reads env vars
from langchain.text_splitter import RecursiveCharacterTextSplitter
#What this does: LangChain's text splitter breaks long paper abstracts into smaller chunks. Why chunk at all? Because embedding models have a token limit — you can't embed an entire paper at once. Smaller chunks also give more precise search results.
from langchain_huggingface import HuggingFaceEmbeddings
#What this does: This is the embedding model. It converts text chunks into vectors (lists of numbers) that capture their meaning. We use HuggingFace's "all-MiniLM-L6-v2" model, which is a popular choice for generating embeddings.
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
#What this does: Qdrant is our vector database. It stores the embeddings and allows us to search for similar vectors later. We use the Qdrant client to connect to our Qdrant instance and manage our vector store.

#block 2 Define the ingest function
COLLECTION_NAME = "ai_trend_digest"

# The embedding model we'll use locally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# This model produces vectors of size 384 (not 1536 like OpenAI)
VECTOR_SIZE = 384

TOPICS = [
    "large language models",
    "diffusion models",
    "reinforcement learning",
    "computer vision transformers",
    "AI agents",
]

PAPERS_PER_TOPIC = 50

#block 3: Fetch papers from arXiv and ingest into Qdrant
def fetch_papers(topic: str, max_results: int) -> list[dict]:
    """Fetch recent papers from arxiv for a given topic."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # most recent first
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "summary": result.summary,      # this is the abstract
            "authors": [a.name for a in result.authors[:5]],
            "published": result.published.strftime("%Y-%m-%d"),
            "url": result.entry_id,         # direct link to the paper
            "topic": topic,
        })
    return papers

# What's happening here: The arxiv Python library gives us a clean interface to arxiv's API. We search by topic, sort by most recent, and pull out the fields we care about. We only take the first 5 authors because some papers have 50+ authors and we don't need all of them in our metadata.

#block 4: build document text
def build_document_text(paper: dict) -> str:
    """Combine paper fields into one string for embedding."""
    return (
        f"Title: {paper['title']}\n\n"
        f"Authors: {', '.join(paper['authors'])}\n"
        f"Published: {paper['published']}\n"
        f"Topic: {paper['topic']}\n\n"
        f"Abstract:\n{paper['summary']}"
    )

#block 5: Main ingest function
def ingest_papers():
    print(" Fetching papers from arxiv...")
    all_papers = []
    for topic in TOPICS:
        print(f"  → {topic}")
        papers = fetch_papers(topic, PAPERS_PER_TOPIC)
        all_papers.extend(papers)
        print(f"     {len(papers)} papers fetched")

    # Save locally so you can inspect what was fetched
    os.makedirs("data", exist_ok=True)
    with open("data/papers.json", "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n {len(all_papers)} papers saved to data/papers.json")

# Why save to JSON? Two reasons: (1) debugging — you can open data/papers.json and see exactly what was fetched, (2) caching — if you need to re-run the embedding step, you don't have to hit the arxiv API again.

# Chunk the documents
    print("\n Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # max characters per chunk
        chunk_overlap=50,  # overlap so context isn't lost at boundaries
    )
    texts, metadatas = [], []
    for paper in all_papers:
        full_text = build_document_text(paper)
        chunks = splitter.split_text(full_text)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "title": paper["title"],
                "url": paper["url"],
                "published": paper["published"],
                "topic": paper["topic"],
                "authors": ", ".join(paper["authors"]),
            })
    print(f" {len(texts)} chunks ready")
    # Set up Qdrant
    print("\n Setting up Qdrant...")
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # Delete old collection if exists, start fresh
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"  → Deleted old collection")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created")
    # Load local embedding model
    print("\n Loading embedding model (downloads once ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use CPU, fine for our size
        encode_kwargs={"normalize_embeddings": True},
    )# Embed and store everything
    print("Embedding and storing in Qdrant (3-5 mins)...")
    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME,
    )
    print(f"{len(texts)} chunks embedded and stored!")
    print("\nIngestion complete! Your vector DB is ready.")


if __name__ == "__main__":
    ingest_papers()
 

