from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def embedding_process() -> (GoogleGenerativeAIEmbeddings | None):
    try:
        print("Embedding running...")
        embedding_model = os.getenv("EMBEDDED_MODEL")
        embeddings: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        return embeddings
    except Exception as error:
        return None