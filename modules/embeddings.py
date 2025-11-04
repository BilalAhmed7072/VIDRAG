from langchain_openai import OpenAIEmbeddings 
from config.config import EMBEDDING_MODEL, OPENAI_API_KEY

embeddings =  OpenAIEmbeddings(api_key=OPENAI_API_KEY , model= EMBEDDING_MODEL)

return embeddings