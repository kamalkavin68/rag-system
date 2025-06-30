import os
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv

from src.services.driver_service import load_document_from_drive
from src.agent.checking_agent import validate_response_with_claude
from src.agent.generative_agent import talk_to_anthropic
from src.services.vector_storage_service import VectorStorageManager
from src.services.chunking_process import SemanticChunkerWithNLP
from src.services.loading_documents import DocumentLoader
from src.services.QueryTransformation import QueryTransformation

load_dotenv()

query_transformer = QueryTransformation()
document_loader = DocumentLoader()
semantic_chunker_nlp = SemanticChunkerWithNLP()
vector_store_manager = VectorStorageManager()

st.set_page_config(page_title="RAG System", layout="wide")
st.title("📄 Retrieval-Augmented Generation (RAG) System")
st.sidebar.title("⚙️ Settings")

defaults = {
    "local_folder_path": "",
    "drive_folder_id": "",
    "document_location": "Local",
    "storage_type": "Old",
    "load_document": False,
    "vector_store": None,
    "chat_history": [],
    "local_data_folder_path": ""
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

st.session_state.document_location = st.sidebar.selectbox("Document Source", ("Local", "Google Drive"))
st.session_state.storage_type = st.sidebar.selectbox("Vector Storage Type", ("Old", "New"))

def handle_folder_selection():
    if st.session_state.document_location == "Local":
        base_path = "vector_storage/local"
    else:
        base_path = "vector_storage/google"

    if st.session_state.storage_type == "Old":
        try:
            folder_list = os.listdir(base_path)
            if folder_list:
                selected = st.sidebar.selectbox("Select Existing Vector Store", folder_list)
                st.session_state.local_folder_path = os.path.join(base_path, selected)
            else:
                st.warning(f"No existing vector stores found in '{base_path}'.")
                st.session_state.local_folder_path = ""
        except FileNotFoundError:
            st.warning(f"Folder '{base_path}' not found.")
            st.session_state.local_folder_path = ""
    else:
        input_key = "local_input" if st.session_state.document_location == "Local" else "drive_input"
        if st.session_state.document_location == "Local":
            st.session_state.local_folder_path = base_path
            st.session_state.local_data_folder_path = st.sidebar.text_input("Enter Local Folder Path", value="", key=input_key)
        else:
            st.session_state.local_folder_path = base_path
            st.session_state.drive_folder_id = st.sidebar.text_input("Enter Google Drive Folder ID", value=st.session_state.drive_folder_id, key=input_key)
handle_folder_selection()


load_disabled = (
    not st.session_state.local_folder_path.strip()
    if st.session_state.document_location == "Local"
    else not st.session_state.drive_folder_id.strip()
)
if st.sidebar.button("Load Documents", disabled=load_disabled):
    st.session_state.load_document = True


def process_documents():
    try:
        with st.spinner("🔄 Loading and indexing documents..."):
            if st.session_state.document_location == "Local":
                if st.session_state.storage_type == "Old":
                    st.session_state.vector_store = vector_store_manager.exist_in_faiss(st.session_state.local_folder_path)
                    st.success("✅ Existing vector store loaded successfully.")
                else:
                    docs = document_loader.load_from_local(st.session_state.local_data_folder_path)
                    chunks = semantic_chunker_nlp.chunk_and_enrich(docs)
                    storage_path = os.path.join(st.session_state.local_folder_path, str(uuid4()))
                    os.makedirs(storage_path, exist_ok=True)
                    st.session_state.vector_store = vector_store_manager.store_in_faiss(chunks, storage_path)
                    st.success(f"✅ Documents processed and indexed successfully from Local folder in folder : {storage_path}")
            else:
                if st.session_state.storage_type == "Old":
                    st.session_state.vector_store = vector_store_manager.exist_in_faiss(st.session_state.local_folder_path)
                    st.success("✅ Loaded vector store from Google Drive.")
                else:
                    docs = load_document_from_drive(st.session_state.drive_folder_id)
                    chunks = semantic_chunker_nlp.chunk_and_enrich(docs)
                    storage_path = os.path.join(st.session_state.local_folder_path, str(uuid4()))
                    os.makedirs(storage_path, exist_ok=True)
                    st.session_state.vector_store = vector_store_manager.store_in_faiss(chunks, storage_path)
                    st.success(f"✅ Documents processed and indexed successfully from Google Drive in folder : {st.session_state.local_folder_path}")
    except Exception as e:
        st.error(f"❌ Failed to load documents: {str(e)}")
    finally:
        st.session_state.load_document = False

if st.session_state.load_document:
    process_documents()

if st.session_state.chat_history:
    for idx, chat in enumerate(st.session_state.chat_history):
        st.chat_message("user").markdown(chat["user_query"])

        with st.expander(f"🧠 View Transformed Queries", expanded=False):
            st.markdown(chat["processed_queries"])

        with st.expander(f"📚 View Retrieved Contexts", expanded=False):
            for i, q in enumerate(chat["contexts"]):
                st.markdown(f"**Q{i+1}:** {q['question']}")
                st.markdown(f"**Context:**\n\n{q['context']}")

        with st.expander(f"🛡️  View Response Validation Result", expanded=False):
            st.markdown(chat["validation"])

        st.chat_message("assistant").markdown(chat["response"])

user_query = st.chat_input("💬 Ask your question here")
if user_query:
    st.chat_message("user").markdown(user_query)

    if not st.session_state.vector_store:
        st.chat_message("assistant").warning("⚠️ Vector store not available. Please load documents first.")
    else:
        try:
            processed_queries = query_transformer.process_query(user_query)

            with st.expander("🧠 View Transformed Queries", expanded=False):
                st.markdown(processed_queries)

            retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
            final_query_context = []

            for key, query in processed_queries.items():
                top_docs = retriever.get_relevant_documents(query)
                context_str = "\n\n".join([doc.page_content for doc in top_docs[:5]])
                final_query_context.append({"question": query, "context": f"[{context_str}]"})

            with st.expander("📚 View Retrieved Contexts", expanded=False):
                for i, q in enumerate(final_query_context):
                    st.markdown(f"**Q{i+1}:** {q['question']}")
                    st.markdown(f"**Context:**\n\n{q['context']}")

            query_context_string = ""
            for idx, item in enumerate(final_query_context):
                query_context_string += f"\nQuestion {idx+1}: {item['question']}\n\nContext {idx+1}:\n{item['context']}\n"

            # Get response and validate
            response, system_prompt = talk_to_anthropic(query_context_string)
            validation_result = validate_response_with_claude(query_context_string, response)

            with st.expander("🛡️ View Response Validation Result", expanded=False):
                st.markdown(validation_result)

            if validation_result != "Valid":
                st.warning("⚠️ Response validation failed. Regenerating...")
                response, _ = talk_to_anthropic(query_context_string)

            # Show assistant response
            st.chat_message("assistant").markdown(response)

            # Save interaction in session history
            st.session_state.chat_history.append({
                "user_query": user_query,
                "processed_queries": processed_queries,
                "contexts": final_query_context,
                "response": response,
                "validation": validation_result
            })

        except Exception as e:
            st.chat_message("assistant").error(f"❌ Error during processing: {str(e)}")
