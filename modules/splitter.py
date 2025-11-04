from langchain_text_splitters import TokenTextSplitter
from config.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(docs):
    splitter= TokenTextSplitter(chunk_size= CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    return chunks