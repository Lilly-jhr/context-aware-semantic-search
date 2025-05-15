# test_sentence_transformer.py
from sentence_transformers import SentenceTransformer
import numpy as np
import os # For cache folder

print("--- Starting test_sentence_transformer.py ---")

print("Attempting to load model...")
cache_dir = os.path.join(os.getcwd(), "model_cache_test_sbert") 
os.makedirs(cache_dir, exist_ok=True)
print(f"Using cache directory: {cache_dir}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"

try:
    # Explicitly load to CPU
    model = SentenceTransformer(model_name, device='cpu', cache_folder=cache_dir)
    print(f"Model '{model_name}' loaded successfully from cache or downloaded.")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("--- Test aborted due to model loading error ---")
    exit()

# Test with a very small number of sentences initially
sentences = [
    "This is sentence one for direct sbert test.", 
    "Here is another sentence for direct sbert testing."
] * 2 # Make it 4 sentences, similar to your 3-doc test

print(f"\nAttempting to embed {len(sentences)} sentences using device: {model.device}")
try:
    embeddings = model.encode(sentences, show_progress_bar=True)
    print("Embeddings generated successfully.")
    print(f"Shape of embeddings: {embeddings.shape}") # Should be (4, 384)
    # print("Sample embedding (first vector):", embeddings[0][:10]) # Print first 10 dims of first embedding
except Exception as e:
    print(f"ERROR during embedding: {e}")

print("\n--- Test_sentence_transformer.py complete ---")