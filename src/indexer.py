import os
import logging
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Using the model chosen based on MTEB and VRAM
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
# Directory to store the persistent Chroma database
# Define path relative to project root
_PROJECT_ROOT_IDX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT_IDX, "chroma_db_index")
# Name for collection within ChromaDB
CHROMA_COLLECTION_NAME = "acordaos"

def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Loads Sentence Transformer embedding model.
    Attempts to use GPU (CUDA) if available, otherwise uses CPU.

    Args:
        model_name: The name of the model to load from Hugging Face.

    Returns:
        The loaded SentenceTransformer model instance.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Loading embedding model '{model_name}' onto device: {device}")
    try:
        model = SentenceTransformer(model_name, device=device)
        logging.info(f"Embedding model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load embedding model '{model_name}': {e}")
        # Maybe consider raising exception or handle it different
        raise

def create_or_update_index(acordao_chunks: List[Dict[str, Any]], model: SentenceTransformer):
    """
    Creates or updates the ChromaDB vector index with given acórdão chunks.

    Use upsert to add new chunks or update existing ones based on generated ID.
    Embeddings are generated using the provided SentenceTransformer model.
    Index is persisted to directory specified by CHROMA_PERSIST_DIR.

    Args:
        acordao_chunks: A list of dictionaries, where each dictionary represents
                        a chunk and contains 'text' and 'metadata' keys.
        model: The loaded SentenceTransformer model instance.
    """
    if not acordao_chunks:
        logging.warning("No acórdão chunks provided to index. Skipping index creation/update.")
        return

    logging.info(f"Starting index creation/update process for {len(acordao_chunks)} chunks...")

    # Initialize ChromaDB Persistent Client
    logging.info(f"Initializing ChromaDB client at path: {CHROMA_PERSIST_DIR}")
    # Ensure the directory exist
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB client at {CHROMA_PERSIST_DIR}: {e}")
        raise # Re-raise exception as we cannot proceed

    # Get or Create Collection
    logging.info(f"Getting or creating collection: {CHROMA_COLLECTION_NAME}")
    try:
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Use cosine distance for E5 models
        )
    except Exception as e:
        logging.error(f"Failed to get or create ChromaDB collection '{CHROMA_COLLECTION_NAME}': {e}")
        raise

    # Prepare Data for ChromaDB
    logging.info("Preparing data for indexing (IDs, prefixed text, metadata)...")
    ids = []
    prefixed_texts = []
    metadatas_list = []
    original_texts = []

    for chunk in acordao_chunks:
        # Create unique and stable ID for each chunk
        metadata = chunk.get('metadata', {})
        source = metadata.get('source', 'unknown_source')
        index = metadata.get('chunk_index', -1)
        chunk_id = f"{os.path.basename(source)}_{index}" # Use basename to keep IDs shorter
        ids.append(chunk_id)

        # Add "passage: " prefix for E5 model embedding generation
        original_text = chunk.get('text', '')
        prefixed_texts.append(f"passage: {original_text}")
        original_texts.append(original_text) # Store original text for the document field

        # Ensure metadata is serializable
        metadatas_list.append(metadata)

    # Check if data preparation yielded results
    if not ids:
         logging.warning("Data preparation resulted in empty lists. Cannot index.")
         return

    # Generate Embeddings
    logging.info(f"Generating embeddings for {len(prefixed_texts)} chunks using model {EMBEDDING_MODEL_NAME}...")
    try:
        # Note: Encoding large batches might consume significant memory (CPU/GPU)
        # Consider batching if memory errors occur for very large input.
        embeddings_list = model.encode(prefixed_texts, show_progress_bar=True)
        # Convert to list if it's a numpy array, ChromaDB expects lists for embeddings
        if not isinstance(embeddings_list, list):
            embeddings_list = embeddings_list.tolist()
        logging.info("Embedding generation complete.")
    except Exception as e:
        logging.error(f"Failed during embedding generation: {e}")
        raise

    # Upsert Data into ChromaDB Collection
    logging.info(f"Upserting {len(ids)} items into ChromaDB collection '{collection.name}'...")
    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=original_texts, # Store original text without prefix
            metadatas=metadatas_list
        )
        logging.info("Upsert completed successfully.")
    except Exception as e:
        logging.error(f"Failed to upsert data into ChromaDB collection '{collection.name}': {e}")
        # Consider logging details like IDs that failed if possible
        raise

# --- Example Usage Block (for direct testing) ---
if __name__ == "__main__":
    print("\n" + "="*20 + " Running indexer.py Direct Execution Test " + "="*20)

    # NOTE: This block needs data from data_loader.
    #       However, we can test model loading here.

    try:
        print(f"Attempting to load model: {EMBEDDING_MODEL_NAME}")
        embedding_model = get_embedding_model()
        print("Model loaded successfully.")
        # Example: Get model device
        print(f"Model is running on device: {embedding_model.device}")

        # Placeholder for calling create_or_update_index if we had sample data here
        # print("\nPlaceholder: Would now call create_or_update_index with actual data.")

    except Exception as e:
        print(f"An error occurred during direct execution test: {e}")

    print("\n" + "="*20 + " End of indexer.py Direct Execution Test " + "="*20) 