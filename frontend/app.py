import sys
import os
import streamlit as st
import logging
import traceback
import shutil


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langchain_core.documents import Document 
from langchain_chroma import Chroma 
from langchain_modules.document_processor import load_documents, split_documents 
from langchain_modules.embedding_manager import get_embedding_function, create_or_update_vector_store
from langchain_modules.query_handler import get_conversational_retrieval_chain 
from langchain_modules.llm_handler import get_llm 
from langchain_modules.nlp_enhancer import extract_entities 
from utils.config import ( 
    COLLECTION_NAME as DEFAULT_COLLECTION_NAME,
    PERSIST_DIRECTORY as DEFAULT_PERSIST_DIRECTORY,
    EMBEDDING_MODEL_NAME as ACTUAL_EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL as DEFAULT_OLLAMA_MODEL,
    OLLAMA_BASE_URL as DEFAULT_OLLAMA_BASE_URL 
)

st.set_page_config(page_title="AI Studio Knowledge Chat", layout="wide", initial_sidebar_state="expanded")
st.title("💡 AI Studio: Chat with Your Knowledge")

logger = logging.getLogger('streamlit_app')
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

default_session_state = {
    'uploaded_file_objects_main': None,
    'chunked_docs_main': [],
    'vector_store_ready_main': False,
    'embedding_model_main': None,
    'llm_main': None,
    'conversational_chain_main': None,
    'chat_history_main': [],
    'vector_store_instance_for_chain_main': None,
    'app_components_loaded_main': False, 
    'processing_complete_main': False 
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Helper Functions (display_chat_messages, initialize_app_components from previous full code)
def display_chat_messages(): 
    for message_data in st.session_state.chat_history_main:
        with st.chat_message(message_data["role"]):
            st.markdown(message_data["content"])
            if message_data["role"] == "assistant" and message_data.get("source_documents"):
                with st.expander("View Sources & Entities", expanded=False):
                    for j, doc in enumerate(message_data["source_documents"]):
                        source = doc.metadata.get('source', 'N/A')
                        page = doc.metadata.get('page')
                        page_display = int(page) + 1 if page is not None else 'N/A'
                        st.markdown(f"**Source {j+1}:** `{source}` (Page: {page_display})")
                        st.caption("Content Preview: " + doc.page_content[:200] + "...")
                        if doc.page_content:
                            entities = extract_entities(doc.page_content) # NER processing
                            if entities:
                                st.markdown("**Key Entities:**")
                                entity_tags = [f"`{ent}` ({lbl})" for lbl, ents in entities.items() for ent in ents]
                                if entity_tags: st.markdown(", ".join(entity_tags))
                                else: st.caption("_No distinct entities._")
                            else: st.caption("_No entities identified._")
                        st.markdown("---")

def initialize_core_models(): 
    """Loads embedding model and LLM if not already loaded. Returns True if both are ready."""
    models_ready = True
    if st.session_state.embedding_model_main is None:
        logger.info("Attempting to load embedding model via initialize_core_models.")
        with st.spinner(f"Loading Embedding Model ({ACTUAL_EMBEDDING_MODEL_NAME})..."):
            try:
                st.session_state.embedding_model_main = get_embedding_function()
                if not callable(getattr(st.session_state.embedding_model_main, 'embed_query', None)):
                    raise ValueError("Embedding model invalid.")
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                st.sidebar.error(f"Embedding Model Load Error: {e}")
                logger.error(f"Embedding model load failed: {e}\n{traceback.format_exc()}")
                st.session_state.embedding_model_main = None
                models_ready = False
    
    if st.session_state.llm_main is None:
        logger.info("Attempting to load LLM via initialize_core_models.")
        with st.spinner(f"Loading LLM ({DEFAULT_OLLAMA_MODEL})... (Ensure Ollama is running)"):
            try:
                st.session_state.llm_main = get_llm() # defaults from config
                if not st.session_state.llm_main:
                    raise ValueError("LLM initialization returned None.")
                logger.info("LLM loaded successfully.")
            except Exception as e:
                st.sidebar.error(f"LLM Load Error: {e}")
                logger.error(f"LLM load failed: {e}\n{traceback.format_exc()}")
                st.session_state.llm_main = None
                models_ready = False
    
    st.session_state.app_components_loaded_main = models_ready 
    return models_ready

# Sidebar
with st.sidebar:
    st.header("⚙️ Global Settings")
    st.subheader("LLM Configuration")
    st.markdown(f"**Model:** `{DEFAULT_OLLAMA_MODEL}` (from config)")
    st.markdown(f"**Base URL:** `{DEFAULT_OLLAMA_BASE_URL}` (from config)")

    st.subheader("Retrieval Configuration")
    st.markdown(f"**Memory Window (K):** `3` (fixed)")
    st.markdown(f"**Retriever Top-K:** `3` (fixed)")  

    st.divider()
    st.header("ℹ️ System Status")
    if st.session_state.embedding_model_main: st.success(f"✓ Embedding Model: `{ACTUAL_EMBEDDING_MODEL_NAME}`")
    else: st.warning("✗ Embedding Model: Not Loaded")
    if st.session_state.llm_main: st.success(f"✓ LLM: `{DEFAULT_OLLAMA_MODEL}`")
    else: st.warning("✗ LLM: Not Loaded")
    if st.session_state.vector_store_ready_main:
         count = 0
         if st.session_state.vector_store_instance_for_chain_main:
            try:
                count = st.session_state.vector_store_instance_for_chain_main._collection.count()
            except Exception:
                pass
         st.success(f"✓ Knowledge Base: Ready (Approx. {count} items)")
    else: st.info("ℹ️ Knowledge Base: Not Built")

    if st.session_state.conversational_chain_main: st.success("✓ Conversational AI: Ready")
    else: st.info("ℹ️ Conversational AI: Not Ready")

    st.divider()
    if st.button("Reset Application State", type="primary", key="reset_app_sidebar_v2"):
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear: del st.session_state[key]
        if os.path.exists(DEFAULT_PERSIST_DIRECTORY):
            try:
                shutil.rmtree(DEFAULT_PERSIST_DIRECTORY)
                st.success(f"Cleared on-disk KB: {DEFAULT_PERSIST_DIRECTORY}")
                os.makedirs(DEFAULT_PERSIST_DIRECTORY, exist_ok=True)
            except Exception as e: st.error(f"Could not delete disk KB: {e}")
        st.rerun()

# load core models on first app load 
if not st.session_state.app_components_loaded_main:
    initialize_core_models() 

# --- Main Panel ---
col1, col2 = st.columns([1, 2]) 

with col1: # document Management 
    st.subheader("📚 Manage Knowledge Base")
    uploaded_files_main = st.file_uploader(
        "1. Upload your documents (PDF, TXT):",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="main_panel_uploader"
    )
    
    st.markdown("2. Configure Processing:")
    chunk_size_main = st.slider("Chunk Size (characters)", 200, 2000, 1000, 100, key="chunk_size_main_panel")
    chunk_overlap_main = st.slider("Chunk Overlap (characters)", 0, 500, 200, 50, key="chunk_overlap_main_panel")

    if st.button("Process Files & Prepare Chat", type="primary", key="process_and_build_button"):
        if not uploaded_files_main:
            st.warning("Please upload files first.")
        elif not st.session_state.embedding_model_main or not st.session_state.llm_main:
            st.error("Core models (Embedding/LLM) not loaded. Check sidebar status and ensure Ollama is running.")
            if not st.session_state.app_components_loaded_main: # load them if button is clicked
                with st.spinner("Models are loading..."):
                    initialize_core_models()
                if not st.session_state.embedding_model_main or not st.session_state.llm_main:
                    st.stop() # Stop if still not loaded
        else:
            with st.spinner("Preparing Knowledge Base: Loading documents..."):
                try:
                    st.session_state.loaded_docs_main = load_documents(uploaded_files_main)
                except Exception as e:
                    st.error(f"Error loading documents: {e}")
                    logger.error(f"Load docs error: {e}\n{traceback.format_exc()}")
                    st.stop()

            with st.spinner("Preparing Knowledge Base: Chunking documents..."):
                try:
                    st.session_state.chunked_docs_main = split_documents(
                        st.session_state.loaded_docs_main,
                        chunk_size=chunk_size_main,
                        chunk_overlap=chunk_overlap_main
                    )
                    st.info(f"Documents processed into {len(st.session_state.chunked_docs_main)} chunks.")
                except Exception as e:
                    st.error(f"Error chunking documents: {e}")
                    logger.error(f"Chunking error: {e}\n{traceback.format_exc()}")
                    st.stop()
            
            with st.spinner("Preparing Knowledge Base: Embedding chunks and updating vector store..."):
                try:
                    vs_instance = create_or_update_vector_store(
                        st.session_state.chunked_docs_main,
                        collection_name=DEFAULT_COLLECTION_NAME,
                        persist_directory=DEFAULT_PERSIST_DIRECTORY,
                        embedding_function=st.session_state.embedding_model_main
                    )
                    st.session_state.vector_store_instance_for_chain_main = Chroma(
                         collection_name=DEFAULT_COLLECTION_NAME,
                         embedding_function=st.session_state.embedding_model_main,
                         persist_directory=DEFAULT_PERSIST_DIRECTORY
                    )
                    count = vs_instance._collection.count()
                    st.success(f"Knowledge Base Ready! Approx. {count} items.")
                    st.session_state.vector_store_ready_main = True
                    st.session_state.processing_complete_main = True # flag 
                    st.session_state.conversational_chain_main = None # force re-creation
                except Exception as e:
                    st.error(f"Error building Knowledge Base: {e}")
                    logger.error(f"KB build error: {e}\n{traceback.format_exc()}")
                    st.session_state.vector_store_ready_main = False
                    st.session_state.processing_complete_main = False
            
            # create chain after KB is ready
            if st.session_state.vector_store_ready_main and st.session_state.llm_main:
                 with st.spinner("Finalizing Chat AI..."):
                    chain = get_conversational_retrieval_chain(
                        llm=st.session_state.llm_main,
                        vector_store=st.session_state.vector_store_instance_for_chain_main
                    )
                    if chain:
                        st.session_state.conversational_chain_main = chain
                        logger.info("Conversational AI prepared after KB build.")
                    else:
                        st.error("Failed to prepare Conversational AI.")


with col2: # Chat 
    st.subheader("💬 Chat with Selected Knowledge")
    
    if not st.session_state.conversational_chain_main and \
       st.session_state.vector_store_ready_main and \
       st.session_state.llm_main and \
       st.session_state.vector_store_instance_for_chain_main: # ensure Chroma instance is loaded
        with st.spinner("Preparing Chat AI..."):
            chain = get_conversational_retrieval_chain(
                llm=st.session_state.llm_main,
                vector_store=st.session_state.vector_store_instance_for_chain_main
            )
            if chain: st.session_state.conversational_chain_main = chain

    if not st.session_state.vector_store_ready_main:
        st.info("☝️ Please upload documents and click 'Process Files & Prepare Chat' to begin.")
    elif not st.session_state.conversational_chain_main:
        st.warning("Conversational AI is not ready. Models might still be loading or KB preparation failed. Check sidebar status.")
    else: # Chat is ready
        display_chat_messages()
        if user_query_chat := st.chat_input("Ask your question here...", key="chat_area_input"):
            st.session_state.chat_history_main.append({"role": "user", "content": user_query_chat})
            with st.chat_message("user"):
                st.markdown(user_query_chat)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                try:
                    response = st.session_state.conversational_chain_main.invoke({
                        "question": user_query_chat
                    })
                    answer = response.get("answer", "Sorry, I couldn't find an answer for that.")
                    source_documents = response.get("source_documents", [])
                    
                    message_placeholder.empty() 
                    
                    # Add to history
                    st.session_state.chat_history_main.append({
                        "role": "assistant",
                        "content": answer,
                        "source_documents": source_documents
                    })
                    st.rerun() 

                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    message_placeholder.empty()
                    st.error(error_msg) 
                    logger.error(f"Error during chat chain invocation: {e}\n{traceback.format_exc()}")
                    st.session_state.chat_history_main.append({"role": "assistant", "content": f"Apologies, an error occurred: {e}", "source_documents": []})
                    st.rerun() 