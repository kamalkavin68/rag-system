import os
import re
import fitz  # PyMuPDF
from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_google_community import GoogleDriveLoader


class DocumentLoader:
    def __init__(
        self,
        credentials_path: str = "secret/credentials.json",
        token_path: str = "secret/token.json",
    ):
        self.credentials_path = credentials_path
        self.token_path = token_path

    def _clean_text(self, text: str) -> str:
        """Cleans extracted text using regular expressions."""
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\n{2,}', '\n\n', text)
        return text.strip()

    def load_from_drive(self, drive_folder_id: str) -> Union[List[Document], None]:
        """Loads PDF documents from a specified Google Drive folder."""
        if not drive_folder_id:
            print("Google Drive folder ID is not provided.")
            return None

        loader = GoogleDriveLoader(
            folder_id=drive_folder_id,
            recursive=False,
            file_types=["pdf"],
            credentials_path=self.credentials_path,
            token_path=self.token_path,
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )

        try:
            documents: List[Document] = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading from Drive: {e}")
            return None

    def load_from_local(self, folder_path: str) -> List[Document]:
        """Loads and cleans text from all PDF files in the specified local folder."""
        if not folder_path or not os.path.isdir(folder_path):
            print(f"Invalid folder path: {folder_path}")
            return []

        cleaned_docs = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(folder_path, filename)
                try:
                    with fitz.open(pdf_path) as doc:
                        for page_num, page in enumerate(doc, start=1):
                            raw_text = page.get_text()
                            cleaned_text = self._clean_text(raw_text)
                            if cleaned_text:
                                cleaned_docs.append(Document(
                                    page_content=cleaned_text,
                                    metadata={
                                        "source": filename,
                                        "page": page_num
                                    }
                                ))
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

        return cleaned_docs
