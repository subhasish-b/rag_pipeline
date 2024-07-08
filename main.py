from fastapi import FastAPI
from typing import Union
from retrival import LoadAndRetrieve
from rag import rag_pipeline

app = FastAPI()

#Initialize the retrieval class
retriever = LoadAndRetrieve()
#Create emdedding and store in db, if not already done
collection = retriever.create_embeddings()
#Initialize the rag pipeline
rag = rag_pipeline()

@app.get("/retrieval")
def retrieve_matching_docs(query : str):
    """
    Handles the retrieval logic, returns top n results along with metadata
    """
    response = retriever.query(collection=collection, query=query)
    return response

@app.get("/rag")
def ask_me(query : str):
    """
    Handles the RAG logic, returns result along with metadata
    """
    context = retriever.query(collection=collection, query=query)
    response = rag.ask(query=query, context=context)
    return response
    
