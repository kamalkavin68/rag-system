from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Iterable

def text_splitter(drive_documents: Iterable[Document]) -> List[Document]:
    print("Splitting running...")
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits: List[Document] = text_splitter.split_documents(drive_documents)
    return splits