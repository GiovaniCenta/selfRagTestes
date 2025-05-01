import os
import logging
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - RETRIEVER - %(levelname)s - %(message)s')

# --- Constants (Should match indexer.py) ---
# Consider refactoring these into a shared config file later
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
# Define path relative to project root
_PROJECT_ROOT_RET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT_RET, "chroma_db_index")
CHROMA_COLLECTION_NAME = "acordaos"
# Prefix for query embeddings (as recommended for E5 models)
QUERY_PREFIX = "query: "

# --- Cached Model and Client ---
# Avoid reloading model and reconnecting to DB for each query if running multiple queries
_embedding_model_instance: Optional[SentenceTransformer] = None
_chroma_client_instance: Optional[chromadb.PersistentClient] = None
_chroma_collection_instance: Optional[chromadb.Collection] = None

def _initialize_retriever(
    model_name: str = EMBEDDING_MODEL_NAME,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME
) -> Tuple[SentenceTransformer, chromadb.Collection]:
    """
    Initializes and caches the embedding model and ChromaDB collection connection.

    Args:
        model_name: Name of the sentence transformer model.
        persist_directory: Path to the ChromaDB persistent storage.
        collection_name: Name of the ChromaDB collection.

    Returns:
        A tuple containing the loaded SentenceTransformer model and the ChromaDB collection.

    Raises:
        Exception: If the model or ChromaDB client/collection fails to initialize.
    """
    global _embedding_model_instance, _chroma_client_instance, _chroma_collection_instance

    # --- Load Embedding Model (Cache if not loaded) ---
    if _embedding_model_instance is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Loading embedding model '{model_name}' onto device: {device}")
        try:
            _embedding_model_instance = SentenceTransformer(model_name, device=device)
            logging.info(f"Embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load embedding model '{model_name}': {e}")
            raise

    # --- Initialize ChromaDB Client (Cache if not connected) ---
    if _chroma_client_instance is None:
        logging.info(f"Initializing ChromaDB client at path: {persist_directory}")
        if not os.path.exists(persist_directory):
             logging.error(f"ChromaDB persist directory not found at {persist_directory}. Run indexer first.")
             raise FileNotFoundError(f"ChromaDB persist directory not found: {persist_directory}")
        try:
            _chroma_client_instance = chromadb.PersistentClient(path=persist_directory)
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB client at {persist_directory}: {e}")
            raise

    # --- Get Collection (Cache if not retrieved) ---
    # Allow re-fetching the collection in case it was deleted/recreated externally
    # if _chroma_collection_instance is None: # <-- Potentially cache this too, but re-getting might be safer
    logging.info(f"Getting collection: {collection_name}")
    try:
        # Check if collection exists before getting
        collections = _chroma_client_instance.list_collections()
        collection_names = [col.name for col in collections]
        if collection_name not in collection_names:
             logging.error(f"Collection '{collection_name}' not found in ChromaDB at {persist_directory}.")
             raise ValueError(f"Collection '{collection_name}' not found. Available: {collection_names}")

        _chroma_collection_instance = _chroma_client_instance.get_collection(
            name=collection_name
            # No need to specify metadata here, it's loaded from persistence
        )
        logging.info(f"Successfully connected to collection '{_chroma_collection_instance.name}' with {_chroma_collection_instance.count()} items.")
    except Exception as e:
        logging.error(f"Failed to get ChromaDB collection '{collection_name}': {e}")
        raise

    return _embedding_model_instance, _chroma_collection_instance


def retrieve_relevant_chunks(
    query_claim: str,
    top_k: int = 3,
    model: Optional[SentenceTransformer] = None,
    collection: Optional[chromadb.Collection] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the top_k most relevant document chunks from ChromaDB for a given query claim.

    Args:
        query_claim: The claim text from the summary to search for.
        top_k: The number of relevant chunks to retrieve. Defaults to 3.
        model: Optional pre-loaded SentenceTransformer model. If None, initializes one.
        collection: Optional pre-loaded ChromaDB collection. If None, initializes one.

    Returns:
        A dictionary containing the query results ('ids', 'documents', 'metadatas', 'distances')
        as returned by chroma_collection.query(), or None if retrieval fails.
        Returns None if the index is empty or retrieval fails.
    """
    if not query_claim:
        logging.warning("Received empty query claim. Cannot retrieve.")
        return None

    try:
        # Initialize model and collection if not provided
        if model is None or collection is None:
            model, collection = _initialize_retriever()

        # Check if the collection is empty
        if collection.count() == 0:
            logging.warning(f"Collection '{collection.name}' is empty. Cannot retrieve.")
            return None

        # 1. Prepare the query with the appropriate prefix
        prefixed_query = f"{QUERY_PREFIX}{query_claim}"
        logging.info(f"Generating embedding for query: '{query_claim[:100]}...'")

        # 2. Generate embedding for the prefixed query
        query_embedding = model.encode([prefixed_query])
        # Ensure it's a list for ChromaDB query
        if not isinstance(query_embedding, list):
            query_embedding = query_embedding.tolist()

        # 3. Query the collection
        logging.info(f"Querying collection '{collection.name}' for top {top_k} results...")
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=['metadatas', 'documents', 'distances'] # Specify what to include
        )
        logging.info(f"Retrieved {len(results.get('ids', [[]])[0])} results.") # Chroma returns lists within lists
        return results

    except FileNotFoundError as e:
        logging.error(f"Retrieval failed: {e}")
        return None
    except ValueError as e: # Catch specific error for collection not found
        logging.error(f"Retrieval failed: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during retrieval: {e}", exc_info=True) # Log traceback
        return None


# --- Example Usage Block (for direct testing) ---
if __name__ == "__main__":
    print("\n" + "="*20 + " Running retriever.py Direct Execution Test " + "="*20)

    # Example query (replace with a real claim from your summaries)
    sample_query = "O BNDES é uma estatal dependente da União"
    print(f'Test Query: "{sample_query}"')
    print(f"Using DB path: {os.path.abspath(CHROMA_PERSIST_DIR)}")
    print(f"Using Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Using Model: {EMBEDDING_MODEL_NAME}")


    try:
        # Direct call - will initialize model and collection internally
        retrieved_results = retrieve_relevant_chunks(sample_query, top_k=3)

        if retrieved_results and retrieved_results.get('ids', [[]])[0]:
            print("\n--- Retrieval Results ---")
            ids = retrieved_results['ids'][0]
            documents = retrieved_results['documents'][0]
            distances = retrieved_results['distances'][0]
            metadatas = retrieved_results['metadatas'][0]

            for i in range(len(ids)):
                print(f"Rank {i+1}:")
                print(f"  ID: {ids[i]}")
                print(f"  Distance: {distances[i]:.4f}")
                print(f"  Metadata: {metadatas[i]}")
                print(f"  Document: {documents[i][:250]}...") # Print preview
                print("-" * 10)
        elif retrieved_results: # Query executed but returned no results
             print("\n--- Retrieval Results ---")
             print("Query executed, but no relevant chunks found.")
        else: # Retrieval failed (error logged)
             print("\n--- Retrieval FAILED ---")
             print("Check logs above for error details.")


    except Exception as e:
        print(f"An error occurred during the direct execution test: {e}")

    print("\n" + "="*20 + " End of retriever.py Direct Execution Test " + "="*20) 