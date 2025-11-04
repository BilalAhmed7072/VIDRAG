from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modules.loader import load_youtube_transcript
from modules.splitter import split_documents
from modules.embeddings import get_openai_embeddings
from modules.vector_store import initialize_pinecone
from modules.retriever import get_retriever
from modules.rag import create_chain
app = FastAPI(title="VidRAG API", description="RAG-powered YouTube Video Q&A API", version="1.0")


class QueryRequest(BaseModel):
    video_url: str
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Given a YouTube video URL and a question, returns an AI-generated answer
    using RAG (Retrieval-Augmented Generation).
    """
    try:
        docs = load_youtube_transcript(request.video_url)
        chunks = split_documents(docs)
        embeddings = get_openai_embeddings()
        vectorstore = initialize_pinecone(embeddings)
        vectorstore.add_documents(chunks)
        retriever = get_retriever(vectorstore)
        rag_chain = create_chain(retriever)
        response = rag_chain.invoke(request.question)

        return {
            "video_url": request.video_url,
            "question": request.question,
            "answer": response.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
