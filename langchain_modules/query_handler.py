import sys
import os
from typing import List, Any, Dict, Tuple
import logging
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langchain_core.documents import Document 
from langchain_core.prompts import PromptTemplate 
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.chains import ConversationalRetrievalChain 
from langchain.memory import ConversationBufferWindowMemory 
from langchain_ollama.llms import OllamaLLM 
from utils.config import ( 
    COLLECTION_NAME as DEFAULT_COLLECTION_NAME,
    PERSIST_DIRECTORY as DEFAULT_PERSIST_DIRECTORY
)
from langchain_modules.llm_handler import get_llm 

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def simple_semantic_search( 
    query: str,
    embedding_function: HuggingFaceEmbeddings,
    k: int = 3,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY
) -> List[Document]:

    if not query: return []
    if not embedding_function: return []
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
        if vector_store._collection.count() == 0: return []
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error during simple semantic search: {e}\n{traceback.format_exc()}")
        return []

# Conversational Chain 
def get_conversational_retrieval_chain(
    llm: OllamaLLM,
    vector_store: Chroma, 
    memory_k: int = 3, 
    return_source_documents: bool = True, 
    k_retriever: int = 3 
):
    """
    Creates and returns a ConversationalRetrievalChain.
    """
    logger.info(f"Creating ConversationalRetrievalChain with memory_k={memory_k}, k_retriever={k_retriever}")
    try:
        memory = ConversationBufferWindowMemory(
            k=memory_k, 
            memory_key="chat_history", 
            return_messages=True, 
            output_key='answer' 
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": k_retriever})

       
        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=return_source_documents, 
            verbose=True 
        )
        logger.info("ConversationalRetrievalChain created successfully.")
        return conversational_chain
    except Exception as e:
        logger.error(f"Error creating ConversationalRetrievalChain: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    logger.info(f"--- Running {__file__} directly for testing conversational chain ---")

    test_llm = None
    test_embed_fn = None
    
    try:
        from langchain_modules.embedding_manager import get_embedding_function
        test_embed_fn = get_embedding_function()
        test_llm = get_llm() 
        if not test_llm or not test_embed_fn:
            raise Exception("LLM or Embedding function failed to initialize.")
        logger.info("LLM and Embedding function loaded for test.")
    except Exception as e:
        logger.error(f"Failed to load LLM/Embedding function for test: {e}")
        sys.exit(1)

    if not os.path.exists(DEFAULT_PERSIST_DIRECTORY) or not os.listdir(DEFAULT_PERSIST_DIRECTORY):
        logger.warning(f"Persist directory '{DEFAULT_PERSIST_DIRECTORY}' is empty or does not exist.")
        logger.warning("Please populate the vector store first.")
        sys.exit(1)

    try:
        test_vector_store = Chroma(
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_function=test_embed_fn,
            persist_directory=DEFAULT_PERSIST_DIRECTORY
        )
        if test_vector_store._collection.count() == 0:
            logger.warning("Vector store collection is empty. Populate it first.")
            sys.exit(1)
        logger.info("Test vector store loaded successfully.")

        chain = get_conversational_retrieval_chain(llm=test_llm, vector_store=test_vector_store)

        if chain:
            logger.info("Conversational chain created. Starting test queries...")
            
            chat_history_sim = [] 
            
            query1 = "What are the main arguments about technology in this document?"
            logger.info(f"\nUser Query 1: {query1}")

            result1 = chain.invoke({"question": query1, "chat_history": chat_history_sim})
            logger.info(f"AI Answer 1: {result1.get('answer')}")
            if result1.get('source_documents'):
                logger.info(f"Source Docs for Q1: {[doc.metadata.get('source') for doc in result1['source_documents']]}")
            


            query2 = "Tell me more about the first point you mentioned."
            logger.info(f"\nUser Query 2 (Follow-up): {query2}")

            result2 = chain.invoke({"question": query2}) 
            logger.info(f"AI Answer 2: {result2.get('answer')}")
            if result2.get('source_documents'):
                 logger.info(f"Source Docs for Q2: {[doc.metadata.get('source') for doc in result2['source_documents']]}")

            query3 = "And what about its impact on solitude?"
            logger.info(f"\nUser Query 3 (Follow-up): {query3}")
            result3 = chain.invoke({"question": query3})
            logger.info(f"AI Answer 3: {result3.get('answer')}")


        else:
            logger.error("Failed to create conversational chain for testing.")

    except Exception as e:
        logger.error(f"Error during conversational chain test: {e}")
        logger.error(traceback.format_exc())

    logger.info(f"--- Finished direct execution of {__file__} ---")