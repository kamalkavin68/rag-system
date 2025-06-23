from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Literal

def vector_Storage(documents: List[Document], embeddings: GoogleGenerativeAIEmbeddings) -> (Chroma | Literal[False]):
    # try:
    print("Vector Storage running...")
    chroma_db_path: str = "./chroma_db"
    vector_store: Chroma = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_db_path
    )
    vector_store.persist()
    return vector_store
    # except Exception as error:
    #     return False
