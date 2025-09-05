import os
from typing import List, Literal, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class VectorStorageManager:
    def __init__(self, embedding_model: Union[GoogleGenerativeAIEmbeddings, None] = None):
        self._embedding_model = embedding_model or GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")
        self._chroma_db_path: str = "./chroma_db"
        self._faiss_index_path: str = "./faiss_index"

    def store_in_chroma(self, documents: List[Document]) -> Union[Chroma, Literal[False]]:
        """
        Stores documents in a Chroma vector store and persists it locally.

        Args:
            documents (List[Document]): List of documents to embed and store.

        Returns:
            Chroma instance if successful, else False.
        """
        try:
            print("Storing vectors in Chroma...")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self._embedding_model,
                persist_directory=self._chroma_db_path
            )
            vector_store.persist()
            return vector_store
        except Exception as e:
            print(f"[ERROR] Chroma storage failed: {e}")
            return False

    def store_in_faiss(self, documents: List[Document], index_path: str) -> Union[FAISS, Literal[False]]:
        """
        Stores documents in a FAISS vector store and saves it locally.

        Args:
            documents (List[Document]): List of documents to embed and store.
            index_path (str): Path to save the FAISS index.

        Returns:
            FAISS instance if successful, else False.
        """
        try:
            print("Storing vectors in FAISS...")
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self._embedding_model
            )
            vector_store.save_local(index_path)
            return vector_store
        except Exception as e:
            print(f"[ERROR] FAISS storage failed: {e}")
            return False

    def exist_in_faiss(self, index_path: str) -> bool:
        """
        Checks if a FAISS index exists at the specified path.

        Returns:
            bool: True if the index exists, else False.
        """
        try:
            vector_store = FAISS.load_local(
                folder_path=index_path,
                embeddings=self._embedding_model,
                allow_dangerous_deserialization=True  # Enable explicit trust
            )
            return vector_store
        except Exception as e:
            print(f"[ERROR] FAISS existence check failed: {e}")
            return False

