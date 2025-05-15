import sys
import os
from typing import List
import logging 
import traceback 


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT) 


from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma 
from utils.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_KWARGS,
    EMBEDDING_ENCODE_KWARGS,
    PERSIST_DIRECTORY,
    COLLECTION_NAME
)

# basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_embedding_function():
    """Initializes and returns the HuggingFace embedding function."""
    logger.info(f"Initializing HuggingFaceEmbeddings with model: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=EMBEDDING_MODEL_KWARGS,
            encode_kwargs=EMBEDDING_ENCODE_KWARGS
        )
        logger.info("HuggingFaceEmbeddings function initialized successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing HuggingFaceEmbeddings: {e}")
        logger.error(traceback.format_exc())
        raise


def create_or_update_vector_store(
    chunked_documents: List[Document],
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = PERSIST_DIRECTORY,
    embedding_function=None
) -> Chroma:
    """
    Creates a new Chroma vector store from documents or updates an existing one.
    Args:
        chunked_documents: A list of LangChain Document objects (chunks).
        collection_name: Name of the collection in Chroma.
        persist_directory: Directory to persist the Chroma database.
        embedding_function: The embedding function to use. If None, initializes one.
    Returns:
        An instance of the Chroma vector store.
    """
    logger.info(f"Attempting to create/update Chroma vector store in '{persist_directory}' with collection '{collection_name}'.")
    if embedding_function is None:
        logger.info("Embedding function not provided, initializing one.")
        embedding_function = get_embedding_function()

    try:
        logger.info("Initializing Chroma client...")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
        logger.info("Chroma client initialized.")
    except Exception as e:
        logger.error(f"Error initializing Chroma client: {e}")
        logger.error(traceback.format_exc())
        raise

    if chunked_documents:
        logger.info(f"Adding {len(chunked_documents)} chunk(s) to Chroma collection '{collection_name}'...")
        try:

            vector_store.add_documents(documents=chunked_documents) 
            
            # Chroma with a persist_directory usually persists automatically.
            count = vector_store._collection.count() 
            logger.info(f"Successfully added documents. Collection '{collection_name}' now has approx. {count} items.")
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            logger.error(traceback.format_exc()) 
            raise
    else:
        logger.warning("No new documents provided to add to the vector store.")

    return vector_store


if __name__ == "__main__":
    logger.info(f"--- Running {__file__} directly for testing ---")
    logger.info(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"Vector store persist directory (from config): {PERSIST_DIRECTORY}")
    logger.info(f"Vector store collection name (from config): {COLLECTION_NAME}")

    # Ensure persist directory exists (config.py should do this, but good for direct test)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    logger.info(f"Ensured persist directory exists: {PERSIST_DIRECTORY}")

    test_embed_fn = None
    try:
        test_embed_fn = get_embedding_function()
    except Exception:
        logger.error("Failed to get embedding function in direct test. Exiting.")
        sys.exit(1)

    sample_docs_for_test = [
        Document(page_content="This is a test document about AI.", metadata={"source": "test_ai.txt", "chunk_num": 1}),
        Document(page_content="LangChain helps build LLM applications.", metadata={"source": "test_langchain.txt", "chunk_num": 1}),
        Document(page_content="Chroma is a vector database.", metadata={"source": "test_chroma.txt", "chunk_num": 1}),
    ]

    logger.info(f"Attempting to create/update vector store with {len(sample_docs_for_test)} sample documents...")
    try:
        db = create_or_update_vector_store(
            sample_docs_for_test,
            embedding_function=test_embed_fn
        )
        if db:
            logger.info("Vector store test completed successfully.")
            count = db._collection.count()
            logger.info(f"Collection '{COLLECTION_NAME}' in '{PERSIST_DIRECTORY}' contains approx. {count} items.")

            # Test similarity search
            if count > 0:
                query = "What is LangChain?"
                logger.info(f"Performing test similarity search for: '{query}'")
                results = db.similarity_search(query, k=2)
                if results:
                    logger.info("Test search results:")
                    for res_doc in results:
                        logger.info(f"  - Content: \"{res_doc.page_content[:60]}...\" (Source: {res_doc.metadata.get('source')})")
                else:
                    logger.info("No results found for test query.")
            else:
                logger.info("Skipping test search as collection is empty or count is zero.")

    except Exception as e:
        logger.error(f"Error during direct test of create_or_update_vector_store: {e}")
        logger.error(traceback.format_exc())
    
    logger.info(f"--- Finished direct execution of {__file__} ---")