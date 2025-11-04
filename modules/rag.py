from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from datetime import datetime

def create_chain(retriever):
    template= """
    You are an AI assistant answering questions based on YouTube video transcripts.
    Use the provided context to generate a factual, timestamped answer.

    Context:
    {context}

    Question:
    {question}

    Include a timestamp: {timestamp}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context","question","timestamp"]
    )

    llm = ChatOpenAI(model="", temperature=0.5)

    runable_map = RunnableMap({
        'context' : retriever,
        'question': RunnablePassthrough(),
        'timestamp': lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    chain = runable_map | prompt | llm
    return chain