import sys
import os
import logging


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langchain_ollama.llms import OllamaLLM 
from utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL 

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_llm(
    model_name: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.1, 
    **kwargs
):
    """
    Initializes and returns an Ollama LLM instance.

    Args:
        model_name (str): The name of the Ollama model to use (e.g., "mistral", "llama2").
        base_url (str): The base URL of the Ollama server.
        temperature (float): Sampling temperature for the LLM.
        **kwargs: Additional keyword arguments to pass to OllamaLLM.

    Returns:
        OllamaLLM instance or None if an error occurs.
    """
    logger.info(f"Initializing OllamaLLM with model='{model_name}', base_url='{base_url}', temperature='{temperature}'")
    try:
        llm = OllamaLLM(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
        
        logger.info("OllamaLLM instance created.")
        return llm
    except Exception as e:
        logger.error(f"Error initializing OllamaLLM: {e}")
        logger.error("Please ensure Ollama server is running and the model '{model_name}' is pulled (e.g., 'ollama pull {model_name}').")
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    logger.info(f"--- Running {__file__} directly for testing LLM connection ---")
    
    llm_instance = get_llm()

    if llm_instance:
        logger.info(f"Successfully initialized LLM with model: {OLLAMA_MODEL}")
        try:
            logger.info("Attempting a test invocation...")
            response = llm_instance.invoke("Why is the sky blue? Respond in one short sentence.")
            logger.info(f"Test LLM Response: {response}")
        except Exception as e:
            logger.error(f"Error during test LLM invocation: {e}")
            logger.error("Ensure Ollama is running and the model is pulled.")
    else:
        logger.error(f"Failed to initialize LLM with model: {OLLAMA_MODEL}")
    
    logger.info(f"--- Finished direct execution of {__file__} ---")