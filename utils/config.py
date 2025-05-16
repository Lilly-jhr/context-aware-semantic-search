import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')

logger.info(f"Attempting to load .env file from: {dotenv_path}")
if os.path.exists(dotenv_path):
    loaded_successfully = load_dotenv(dotenv_path=dotenv_path, override=True)
    if loaded_successfully:
        logger.info(f".env file loaded successfully from {dotenv_path}.")
    else:
        logger.warning(f"Found .env file at {dotenv_path}, but python-dotenv indicated it failed to load variables (or it was empty).")
else:
    logger.warning(f".env file not found at {dotenv_path}. Using default configurations or environment variables.")

# Embedding Model Configuration
DEFAULT_FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_FALLBACK_EMBEDDING_MODEL)
logger.info(f"EMBEDDING_MODEL_NAME set to: {EMBEDDING_MODEL_NAME}")
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': False}

# Vector Store Configuration
VECTOR_STORE_DATA_PATH = os.path.join(BASE_DIR, "data", "vector_stores")
PERSIST_DIRECTORY = os.path.join(VECTOR_STORE_DATA_PATH, "chroma_db_main_lc")
DEFAULT_FALLBACK_COLLECTION_NAME = "knowledge_base_default_collection"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", DEFAULT_FALLBACK_COLLECTION_NAME)
logger.info(f"COLLECTION_NAME set to: {COLLECTION_NAME}")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral") 
logger.info(f"OLLAMA_BASE_URL set to: {OLLAMA_BASE_URL}")
logger.info(f"OLLAMA_MODEL set to: {OLLAMA_MODEL}")