def get_retriever (vectorstore, top_k=2):
    retriever = vectorstore.as_retriever(search_type= "similarity",search_kwargs={"k":top_k})
    return retriever