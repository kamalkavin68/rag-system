from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List

def vector_retrieval(vector_store:Chroma, user_query: str) -> List[Document]:
    print("Vector Retrieval running...")
    query: str = user_query
    retriever: VectorStoreRetriever = vector_store.as_retriever(search_type="similarity", search_kwargs ={"k": 20})
    results: List[Document] = retriever.get_relevant_documents(query)
    return results