import streamlit as st
from dotenv import load_dotenv
from src.agent.generative_agent import talk_to_anthropic
from src.runner.rag_runner import rag_runner_fun, retriever_runner_fun
load_dotenv() 

from typing import List
from langchain_core.documents import Document

from src.services.driver_service import load_document_from_drive


st.title("Google Drive RAG")

st.sidebar.header("Chat Settings")

if "drive_documents" not in st.session_state:
    st.session_state.drive_documents = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

google_drive_folder_id: str = st.sidebar.text_input("Google Drive Folder ID", key="folder_id_input")

is_submit_clicked: bool = st.sidebar.button("Load Documents", key="submit_button")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if is_submit_clicked:
    if google_drive_folder_id:
        with st.spinner("Loading documents from Google Drive... Please wait."):
        
            st.session_state.drive_documents = load_document_from_drive(google_drive_folder_id)

            st.session_state.vector_store = rag_runner_fun(st.session_state.drive_documents, st)

        if st.session_state.drive_documents:
            
            st.info("You can now proceed with your RAG operations.")
        else:
            st.error("Error in Document Load from Google Drive")
            st.info("Ensure the Google Drive Folder ID is correct and that the service account has the necessary access.")
    else:
        st.warning("Please enter the Google Drive Folder ID and click 'Load Documents'.")

if st.session_state.drive_documents:
    prompt = st.chat_input("What is up?")
    if prompt:
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            context : str = retriever_runner_fun(st.session_state.vector_store, prompt, st)
            chat_history = st.session_state.messages
            response: str = talk_to_anthropic(prompt, chat_history , context)
        
        st.session_state.messages.append({"role": "user", "content": f"{prompt}"})
            
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response}) 
else:
    st.info("Please load documents from Google Drive to start chatting.")


st.sidebar.markdown(
    """
    ---
    **How to find your Google Drive Folder ID:**
    1. Go to your Google Drive in a web browser.
    2. Navigate to the folder you want to use.
    3. The Folder ID is the long string of characters in the URL after `https://drive.google.com/drive/folders/`.
    """
)