#block 1
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
#the rag chain needs the same embedding model as ingestion, when the student asks a question, we convert their question to a vector using the exact model we used to embed papers
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

#we are connecting qdrant again but as a reader, not writer

from langchain_groq import ChatGroq
#this will generate the actual answer after we retrieve the relevant papers

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#retrivalqa is langchain prebuilt chain that connects retrieval and generation in one call
#give it a question, it fetches relevant chunks from qdrant and passes it to llm and returns an answer

#prompttemplate lets us write a custom prompt to that the llm knows its talking to students and should explain things with citations

#block2:config
COLLECTION_NAME = "ai_trend_digest"
EMBEDDING_MODEL='all-miniLM-L6-v2'

#block 3: load the vector store
def load_vectorstore():
    "Connect to Qdrant and load the vector store for retrieval."
    embeddings=HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
#we are loading huggingface model into memory, since we have already downloaded it during ingestion, this is instant, no download needed
    client=QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    #why create client seperately? we will directly pass it to QdrantVectorStore, we have full control over the connection, makes the testing independent
    vectorstore=QdrantVectorStore(
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME
    )
    return vectorstore
    # this returs a langchain object that wraps the qdrant collection, it knows how to take a ny text, embed it and search for most similar chunks, think of it as a smart search engine over your papers
   
   #block 4: build the prompt

def build_prompt():
    """Create a student-friendly prompt template."""
    template = """
You are an AI research assistant helping students learn about recent AI trends.

Use the following research paper excerpts to answer the question.
Explain concepts clearly and simply — assume the student understands ML basics
but may not be familiar with cutting-edge research.

Always:
- Cite the paper titles you used in your answer
- Explain any technical terms you use
- If the papers don't contain enough info to answer, say so honestly

Context from papers:
{context}

Student's question: {question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

#block 5: build the full rag chain
def build_rag_chain():
    """Wire together retrieval + generation into one chain."""
    print(" Loading vector store...")
    vectorstore = load_vectorstore()

    print("🤖 Loading Groq LLM...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",    # fast, free, great for Q&A
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,            # low = more factual, less creative
    )
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5}  )
    prompt=build_prompt()
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", #stuff means we put chunks into one prompt
        retriever=retriever,
        return_source_documents=True, #also returns which papers were used
        chain_type_kwargs={"prompt":prompt}
    )
    return chain

def query(question: str, chain) -> dict:
    """Run a question through the RAG chain and return answer + sources."""
    result = chain.invoke({"query": question})

    # Extract unique source papers
    sources = []
    seen = set()
    for doc in result["source_documents"]:
        title = doc.metadata.get("title", "Unknown")
        if title not in seen:
            seen.add(title)
            sources.append({
                "title": title,
                "url": doc.metadata.get("url", ""),
                "published": doc.metadata.get("published", ""),
                "topic": doc.metadata.get("topic", ""),
            })

    return {
        "answer": result["result"],
        "sources": sources,
    }

# What's happening here: After the chain runs, result["result"] is the LLM's answer and result["source_documents"] is the list of chunks it used. We deduplicate by title (since multiple chunks can come from the same paper) and return a clean list of source papers the student can actually go read.

if __name__ == "__main__":
    print("Building RAG chain...")
    chain = build_rag_chain()

    test_question = "What are the latest trends in large language models?"
    print(f"\n ---QUESTION: {test_question}---\n")

    result = query(test_question, chain)

    print("---ANSWER---:")
    print(result["answer"])

    print("\n ---SOURCES USED---:")
    for s in result["sources"]:
        print(f"  - {s['title']} ({s['published']})")
        print(f"    {s['url']}")