import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from langchain_modules.document_processor import load_documents, split_documents


st.set_page_config(
    page_title="Context-Aware Semantic Search Platform - Ingestion",
    layout="wide"
)

st.title("📚 Document Ingestion and Chunking")
st.write("""
Upload your knowledge base documents (PDFs, TXT files).
They will be loaded, processed, and split into manageable chunks.
""")

# --- Session State Initialization ---
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []
if 'loaded_docs' not in st.session_state:
    st.session_state.loaded_docs = []
if 'chunked_docs' not in st.session_state:
    st.session_state.chunked_docs = []
if 'processed_filenames' not in st.session_state:
    st.session_state.processed_filenames = set()


# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload PDF or TXT documents:",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="You can upload multiple files at once."
)

# --- Document Processing Parameters ---
st.sidebar.header("Chunking Parameters")
chunk_size = st.sidebar.slider("Chunk Size (characters)", 200, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", 0, 500, 200, 50)

# --- Process Uploaded Files ---
if uploaded_files:
    new_files_to_process = []
    for up_file in uploaded_files:
        if up_file.name not in st.session_state.processed_filenames:
            new_files_to_process.append(up_file)
            st.session_state.processed_filenames.add(up_file.name) # Add to processed set

    if new_files_to_process:
        st.info(f"Processing {len(new_files_to_process)} new file(s)... This may take a moment.")
        
        # Append new files to the session state list
        st.session_state.uploaded_files_list.extend(new_files_to_process)

        # Load new documents
        with st.spinner("Loading documents..."):
            newly_loaded_docs = load_documents(new_files_to_process)
            st.session_state.loaded_docs.extend(newly_loaded_docs)
        
        st.success(f"Successfully loaded {len(newly_loaded_docs)} new document(s)/page(s).")
        st.write(f"Total documents/pages loaded so far: {len(st.session_state.loaded_docs)}")

        # Always re-chunk all loaded documents if parameters or files change
        if st.session_state.loaded_docs:
            with st.spinner(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})..."):
                st.session_state.chunked_docs = split_documents(
                    st.session_state.loaded_docs, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
            st.success(f"Successfully split documents into {len(st.session_state.chunked_docs)} chunks.")
    elif not st.session_state.uploaded_files_list: # No files ever uploaded
        st.warning("No files uploaded yet.")
    else: # Files were previously uploaded, but no new ones in this run
        st.info("Previously uploaded files are processed. You can upload more or adjust chunking parameters.")
        # Re-chunk if parameters changed but no new files
        if st.session_state.loaded_docs and (
            st.session_state.get('last_chunk_size') != chunk_size or
            st.session_state.get('last_chunk_overlap') != chunk_overlap
        ):
            with st.spinner(f"Re-splitting documents with new parameters (size: {chunk_size}, overlap: {chunk_overlap})..."):
                st.session_state.chunked_docs = split_documents(
                    st.session_state.loaded_docs, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
            st.success(f"Re-split documents into {len(st.session_state.chunked_docs)} chunks with new parameters.")


    # Store current chunking parameters to detect changes
    st.session_state.last_chunk_size = chunk_size
    st.session_state.last_chunk_overlap = chunk_overlap

# --- Display Processed Information ---
if st.session_state.chunked_docs:
    st.subheader("Processed Chunks Overview")
    
    total_docs = len(st.session_state.loaded_docs)
    total_chunks = len(st.session_state.chunked_docs)
    
    st.metric(label="Total Documents/Pages Loaded", value=total_docs)
    st.metric(label="Total Chunks Created", value=total_chunks)

    st.write("Sample Chunks:")
    # Create a list of (source, page, content) for display
    sample_chunks_display = []
    for i, chunk in enumerate(st.session_state.chunked_docs[:5]): # Display first 5 chunks
        source = chunk.metadata.get('source', 'N/A')
        page = chunk.metadata.get('page', 'N/A')
        content_preview = chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
        
        expander_title = f"Chunk {i+1} (Source: {source}, Page: {page if page != 'N/A' else 'N/A'})"
        with st.expander(expander_title):
            st.markdown(f"**Source:** `{source}`")
            if page != 'N/A': # Only show page if it's relevant (i.e., from PDF)
                st.markdown(f"**Page:** `{page + 1}` (0-indexed is {page})") # Display 1-indexed page for users
            st.markdown(f"**Start Index in Original Doc:** `{chunk.metadata.get('start_index', 'N/A')}`")
            st.markdown("**Content Preview:**")
            st.markdown(f"> {content_preview}")
            # st.text_area("Full Chunk Content:", chunk.page_content, height=150, key=f"chunk_content_{i}")


# --- Button to Clear ---
if st.sidebar.button("Clear All Processed Data"):
    st.session_state.uploaded_files_list = []
    st.session_state.loaded_docs = []
    st.session_state.chunked_docs = []
    st.session_state.processed_filenames = set()
    st.success("Cleared all processed document data. You can upload new files.")
    st.rerun()

st.sidebar.info(
    "This application demonstrates document ingestion and chunking. "
    "In the next steps, these chunks will be embedded and stored in a vector database for semantic search."
)