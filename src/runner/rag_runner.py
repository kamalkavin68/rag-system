
from typing import List, Any, Literal
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import streamlit

from src.services.retrivel_service import vector_retrieval
from src.services.vector_storage_service import vector_Storage
from src.agent.emdedding_agent import embedding_process
from src.services.text_splitter_service import text_splitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def rag_runner_fun(drive_documents: List[Document], st: streamlit) -> Chroma:
    doc_set = set()
    for doc in drive_documents:
        content: str = doc.page_content
        source: str = doc.metadata.get('source')
        mime_type: str = doc.metadata.get('mimeType')
        title: str = doc.metadata.get('title')
        doc_set.add(title)

    st.info(f"{len(doc_set)} documents loaded successfully from Google Drive")

    documents: List[Document] = text_splitter(drive_documents)
    st.success("Documents Splitted successfully")

    embeddings: GoogleGenerativeAIEmbeddings = embedding_process()
    st.success("Embeddings Generated successfully")


    vector_store: Chroma = vector_Storage(documents, embeddings)
    st.success("Vector Store Created successfully")

    return vector_store

def retriever_runner_fun(vector_store: Chroma, user_query: str, st: streamlit):
    # user_query = "What is this document about?"
    results: List[Document] = vector_retrieval(vector_store, user_query)
    st.success("Vector Store Retrieved successfully")

    results: str = "\n\n".join([doc.page_content for doc in results])

    return results