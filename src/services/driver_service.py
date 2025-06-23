from langchain_google_community import GoogleDriveLoader
from langchain_core.documents import Document
from typing import List

def load_document_from_drive(folder_id) -> (List[Document] | None):
    """Load documents from a Google Drive folder"""

    # try:
    loader: GoogleDriveLoader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=False,
        file_types=["pdf"],
        credentials_path="secret/credentials.json",
        token_path="secret/token.json",
        scopes=["https://www.googleapis.com/auth/drive.readonly"] 
    )
    documents: List[Document] = loader.load()
    return documents

    # except Exception as error:
    #     print(error)
    #     return None
