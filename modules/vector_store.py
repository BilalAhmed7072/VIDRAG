from langchain_pinecone import PineconeVectorStore
from config.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from pinecone import Pinecone , ServerlessSpec

def initialize_pinecone (embeddings):
    pc = Pinecone (api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes().indexes]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec = ServerlessSpec(cloud="",region="")
        )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore