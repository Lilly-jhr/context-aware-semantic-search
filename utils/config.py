import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=dotenv_path)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'} # or 'cuda'
EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': False}

VECTOR_STORE_DATA_PATH = os.path.join(BASE_DIR, "data", "vector_stores")

PERSIST_DIRECTORY = os.path.join(VECTOR_STORE_DATA_PATH, "chroma_db_main_lc") 
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base_collection_lc") 

os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")