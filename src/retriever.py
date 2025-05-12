import os
import logging
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Tuple, Optional

# Assuming access to the embedding model function from data_loader
from .data_loader import get_embedding_model 
# Corrected import: Removed _get_cosmos_container
from .indexer import _get_chroma_collection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - RETRIEVER - %(levelname)s - %(message)s')

# --- Constants (Should match indexer.py) ---
# Consider refactoring these into a shared config file later
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
# Define path relative to project root
_PROJECT_ROOT_RET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT_RET, "chroma_db_index")
# Nome da coleção - usa o nome disponível na base
CHROMA_COLLECTION_NAME = "acordaos"  # Corrigido: era acordaos_legal_bertimbau
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


def search(query: str, processo_id: str, k: int = 20) -> List[Dict]:
    """
    Performs a vector search in ChromaDB, filtered by 'processo_id',
    to retrieve the top k relevant document chunks.

    Args:
        query: The search query text (claim).
        processo_id: The process ID to filter results (must match metadata in ChromaDB).
        k: The maximum number of results to retrieve.

    Returns:
        A list of document dictionaries, each containing at least 'id', 'text',
        'metadata' (including 'processo'), and 'distance' (or 'score').
        Returns an empty list if an error occurs or no results are found.
    """
    if not query or not processo_id:
        logging.warning("Query and processo_id must be provided for ChromaDB search.")
        return []

    try:
        # 1. Get Query Vector using the same model as indexing
        logging.debug(f"Generating embedding for query (retriever): '{query[:50]}...'")
        embedder = get_embedding_model() # Uses cached model from data_loader
        # Ensure normalization matches indexing if it was done (pdf_to_docs does normalize_embeddings=True)
        query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
        logging.debug(f"Query vector generated for ChromaDB search (dim: {len(query_vector)}).")

        # 2. Get ChromaDB Collection
        collection = _get_chroma_collection() # Uses cached collection from indexer

        if collection.count() == 0:
            logging.warning(f"ChromaDB collection '{collection.name}' is empty. Cannot retrieve.")
            return []

        # 3. Perform Vector Search with Metadata Filter for 'processo_id'
        logging.info(f"Querying ChromaDB collection '{collection.name}' for top {k} results "
                     f"where processo = '{processo_id}'...")
        
        results = collection.query(
            query_embeddings=[query_vector], # Chroma expects a list of embeddings
            n_results=k,
            where={"processo": processo_id}, # Metadata filter
            include=['metadatas', 'documents', 'distances'] # Specify what to include
        )
        
        retrieved_docs = []
        if results and results.get('ids') and results.get('ids')[0]: # Chroma returns lists within lists
            ids_list = results['ids'][0]
            documents_list = results['documents'][0]
            metadatas_list = results['metadatas'][0]
            distances_list = results['distances'][0]
            
            for i in range(len(ids_list)):
                doc = {
                    "id": ids_list[i],
                    "text": documents_list[i],
                    "metadata": metadatas_list[i],
                    "distance": distances_list[i] # Cosine distance, lower is better
                    # If you prefer a score (0-1, higher is better) for cosine:
                    # "score": 1 - distances_list[i] if distances_list[i] is not None else 0 
                }
                retrieved_docs.append(doc)
            logging.info(f"Retrieved {len(retrieved_docs)} candidate documents from ChromaDB for processo '{processo_id}'.")
        else:
            logging.info(f"No results found in ChromaDB for processo '{processo_id}' matching the query.")
            
        return retrieved_docs

    except Exception as e:
        logging.error(f"Failed to retrieve documents from ChromaDB for query '{query[:50]}...' and processo '{processo_id}': {e}", exc_info=True)
        return []

# --- Example Usage Block (for direct testing) ---
if __name__ == "__main__":
    print("\n" + "="*20 + " Running retriever.py (ChromaDB) Direct Execution Test " + "="*20)

    # Ensure CHROMA_PERSIST_DIR exists for the test to run meaningfully.
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"ChromaDB persistence directory NOT FOUND: {os.path.abspath(CHROMA_PERSIST_DIR)}")
        print("Please run the indexer (e.g., python src/indexer.py) first to create and populate the DB.")
    else:
        print(f"Using ChromaDB from: {os.path.abspath(CHROMA_PERSIST_DIR)}")
        print(f"Querying collection: {CHROMA_COLLECTION_NAME}")

        # Example query and processo_id (ensure these exist in your test ChromaDB)
        test_query = "primeiro parágrafo do processo um" # Query that might match dummy data in indexer.py
        test_processo_id = "PROC1-2024" # Must match a 'processo' value in metadata
        
        print(f"\nTesting retrieval for query: '{test_query}'")
        print(f"Filtering for processo: '{test_processo_id}'")
        
        retrieved_documents = search(test_query, test_processo_id, k=2)
        
        if retrieved_documents:
            print(f"\nSuccessfully retrieved {len(retrieved_documents)} documents:")
            for i, doc_info in enumerate(retrieved_documents):
                print(f"  Rank {i+1}: ID={doc_info.get('id')}, Distance={doc_info.get('distance'):.4f}")
                print(f"    Text: {doc_info.get('text', '')[:100]}...")
                print(f"    Metadata: {doc_info.get('metadata')}")
        else:
            print("\nFailed to retrieve documents or no documents found for the given query and processo.")
            print("Ensure data for the specified processo_id exists in your ChromaDB collection and the indexer was run.")

    print("\n" + "="*60 + "\n") 