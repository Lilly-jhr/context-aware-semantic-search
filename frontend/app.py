import sys
import os
import streamlit as st
import logging 
import traceback 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT) 

from langchain_core.documents import Document 
from langchain_modules.document_processor import load_documents, split_documents 
from langchain_modules.embedding_manager import get_embedding_function, create_or_update_vector_store 
from utils.config import COLLECTION_NAME as DEFAULT_COLLECTION_NAME, PERSIST_DIRECTORY as DEFAULT_PERSIST_DIRECTORY 

st.set_page_config(page_title="Semantic Search Platform - Ingestion & Embedding", layout="wide")
st.title("📚 Semantic Search Platform")

logger = logging.getLogger('streamlit_app')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Session State Initialization 
if 'loaded_docs' not in st.session_state:
    st.session_state.loaded_docs = []
if 'chunked_docs' not in st.session_state:
    st.session_state.chunked_docs = []
if 'processed_filenames_for_chunking' not in st.session_state: # tracks files whose content is in loaded_docs/chunked_docs
    st.session_state.processed_filenames_for_chunking = set()
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'embedding_model' not in st.session_state: # cache the embedding mode
    st.session_state.embedding_model = None


# UI
st.header("1. Document Upload & Chunking")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT documents:", type=["pdf", "txt"], accept_multiple_files=True,
    help="Upload new documents or re-upload to re-process."
)

st.sidebar.header("Processing Parameters")
chunk_size_val = st.sidebar.slider("Chunk Size", 200, 2000, 1000, 100, key="chunk_size")
chunk_overlap_val = st.sidebar.slider("Chunk Overlap", 0, 500, 200, 50, key="chunk_overlap")

# re-process if files change or parameters change

unique_uploaded_file_key = ""
if uploaded_files:
    unique_uploaded_file_key = "_".join(sorted([f.name for f in uploaded_files])) + f"_{chunk_size_val}_{chunk_overlap_val}"

if 'last_processed_key' not in st.session_state:
    st.session_state.last_processed_key = ""

if uploaded_files and unique_uploaded_file_key != st.session_state.get('last_processed_key_chunking'):
    st.session_state.last_processed_key_chunking = unique_uploaded_file_key
    st.info("New files or parameters detected. Processing documents...")
    with st.spinner("Loading and chunking documents..."):
        try:
            st.session_state.loaded_docs = load_documents(uploaded_files) # uploaded_files is list of Streamlit UploadedFile
            st.session_state.chunked_docs = split_documents(
                st.session_state.loaded_docs,
                chunk_size=chunk_size_val,
                chunk_overlap=chunk_overlap_val
            )
            st.session_state.processed_filenames_for_chunking = {f.name for f in uploaded_files}
            st.success(f"Processed {len(st.session_state.loaded_docs)} pages/docs into {len(st.session_state.chunked_docs)} chunks.")
            st.session_state.vector_store_ready = False # chunks mean vector store needs update
        except Exception as e:
            st.error(f"Error during document processing: {e}")
            logger.error(f"Document processing error: {e}\n{traceback.format_exc()}")
            st.session_state.chunked_docs = [] # Clear chunks on error
elif not uploaded_files and st.session_state.chunked_docs:
     # If files are removed from uploader, clear the chunks
     st.session_state.chunked_docs = []
     st.session_state.loaded_docs = []
     st.session_state.processed_filenames_for_chunking = set()
     st.session_state.vector_store_ready = False
     st.info("File uploader is empty. Cleared previously processed chunks.")


if st.session_state.chunked_docs:
    st.metric(label="Total Chunks Ready for Embedding", value=len(st.session_state.chunked_docs))
    with st.expander("View Sample Chunks (First 3)"):
        for i, chunk in enumerate(st.session_state.chunked_docs[:3]):
            st.markdown(f"**Chunk {i+1} (Source: {chunk.metadata.get('source', 'N/A')}, Page: {chunk.metadata.get('page', 'N/A')})**")
            st.caption(chunk.page_content[:250] + "...")
else:
    st.info("Upload documents and they will be processed into chunks here.")

st.divider()
st.header("2. Embedding and Vector Store ")

# Embedding model is loaded once and cached in session state
if st.session_state.embedding_model is None:
    with st.spinner("Initializing embedding model... (May take time on first run)"):
        try:
            st.session_state.embedding_model = get_embedding_function()
            st.success("Embedding model loaded and cached in session.")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            logger.error(f"Embedding model loading error: {e}\n{traceback.format_exc()}")
            st.stop()
else:
    DEFAULT_EMBEDDING_MODEL_NAME = "default-embedding-model"  
    st.info(f"Using cached embedding model: {DEFAULT_EMBEDDING_MODEL_NAME}")

if st.session_state.chunked_docs and st.session_state.embedding_model:
    if st.button("Create/Update Vector Store with Current Chunks", key="embed_button"):
        with st.spinner(f"Embedding {len(st.session_state.chunked_docs)} chunks and updating Chroma vector store..."):
            try:
               
                vector_store_instance = create_or_update_vector_store(
                    st.session_state.chunked_docs,
                    collection_name=DEFAULT_COLLECTION_NAME,
                    persist_directory=DEFAULT_PERSIST_DIRECTORY,
                    embedding_function=st.session_state.embedding_model
                )
                count = vector_store_instance._collection.count()
                st.success("Vector store updated!")
                st.session_state.vector_store_ready = True
            except Exception as e:
                st.error(f"Error updating vector store: {e}")
                logger.error(f"Vector store update error (from Streamlit): {e}\n{traceback.format_exc()}")
                st.session_state.vector_store_ready = False
elif not st.session_state.chunked_docs:
    st.warning("No document chunks available. Please upload and process documents first.")
elif not st.session_state.embedding_model:
    st.warning("Embedding model not loaded. Cannot proceed.")


if st.session_state.vector_store_ready:
    st.success(f"Vector store is ready. Collection: '{DEFAULT_COLLECTION_NAME}'.")
else:
    st.info("Vector store not yet created/updated in this session with current chunks.")

st.sidebar.divider()
if st.sidebar.button("Full Reset (Clear Session & Attempt Disk Clear)", type="primary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    import shutil
    if os.path.exists(DEFAULT_PERSIST_DIRECTORY):
        try:
            shutil.rmtree(DEFAULT_PERSIST_DIRECTORY)
            st.sidebar.success(f"Cleared on-disk vector store at {DEFAULT_PERSIST_DIRECTORY}")
            os.makedirs(DEFAULT_PERSIST_DIRECTORY, exist_ok=True) # Recreate
        except Exception as e:
            st.sidebar.error(f"Could not delete disk store: {e}")
    else:
        st.sidebar.info("No on-disk store found at specified path to clear.")
    
    st.rerun()

st.sidebar.info(f"Vector Store Path: {DEFAULT_PERSIST_DIRECTORY}")
st.sidebar.info(f"Collection Name: {DEFAULT_COLLECTION_NAME}")