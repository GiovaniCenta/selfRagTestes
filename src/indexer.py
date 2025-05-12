import os
import logging
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - INDEXER - %(levelname)s - %(message)s')

# --- ChromaDB Constants ---
_PROJECT_ROOT_IDX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT_IDX, "chroma_db_index")
CHROMA_COLLECTION_NAME = "acordaos" 
EMBEDDING_MODEL_DIMENSION = 1024 
EMBEDDING_MODEL_NAME_FOR_CHROMA_META = "rufimelo/Legal-BERTimbau-sts-large" 

# --- ChromaDB Client (Cached) ---
_chroma_client_instance: Optional[chromadb.PersistentClient] = None
_chroma_collection_instance: Optional[chromadb.Collection] = None

def _get_chroma_collection() -> chromadb.Collection:
    """Initializes and returns a singleton ChromaDB collection instance."""
    global _chroma_client_instance, _chroma_collection_instance

    if _chroma_collection_instance is not None:
        return _chroma_collection_instance

    if _chroma_client_instance is None:
        logging.info(f"Initializing ChromaDB PersistentClient at path: {CHROMA_PERSIST_DIR}")
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        try:
            _chroma_client_instance = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB client at {CHROMA_PERSIST_DIR}: {e}", exc_info=True)
            raise
    
    logging.info(f"Getting or creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
    try:
        _chroma_collection_instance = _chroma_client_instance.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine", 
                "embedding_model": EMBEDDING_MODEL_NAME_FOR_CHROMA_META,
                "dimension": EMBEDDING_MODEL_DIMENSION
            }
        )
        logging.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready with {_chroma_collection_instance.count()} items.")
    except Exception as e:
        logging.error(f"Failed to get or create ChromaDB collection '{CHROMA_COLLECTION_NAME}': {e}", exc_info=True)
        raise
    
    return _chroma_collection_instance

def create_or_update_index_chroma(docs_with_embeddings: List[Dict[str, Any]]):
    """
    Creates or updates the ChromaDB vector index with documents that already have embeddings.
    Input dicts require 'id', 'text', 'vector', 'processo'.
    """
    if not docs_with_embeddings:
        logging.warning("No documents provided to index. Skipping index creation/update.")
        return

    collection = _get_chroma_collection()
    
    ids_list = []
    embeddings_list = []
    documents_list = [] 
    metadatas_list = [] 

    logging.info(f"Preparing {len(docs_with_embeddings)} documents for ChromaDB upsert...")
    for doc in docs_with_embeddings:
        if not all(k in doc for k in ['id', 'text', 'vector', 'processo']):
            logging.warning(f"Document missing required keys (id, text, vector, processo). Skipping: {doc.get('id', 'UNKNOWN_ID')}")
            continue

        ids_list.append(doc['id'])
        embeddings_list.append(doc['vector'])
        documents_list.append(doc['text'])
        
        metadata = {key: value for key, value in doc.items() if key not in ['id', 'text', 'vector']}
        metadata['processo'] = doc['processo'] 
        metadatas_list.append(metadata)

    if not ids_list:
        logging.warning("No valid documents to upsert after preparation. Skipping.")
        return

    logging.info(f"Upserting {len(ids_list)} items into ChromaDB collection '{collection.name}'...")
    try:
        collection.upsert(
            ids=ids_list,
            embeddings=embeddings_list,
            documents=documents_list,
            metadatas=metadatas_list
        )
        logging.info(f"Upsert completed. Collection now has {collection.count()} items.")
    except Exception as e:
        logging.error(f"Failed to upsert data into ChromaDB collection '{collection.name}': {e}", exc_info=True)
        raise

upsert_items = create_or_update_index_chroma

if __name__ == '__main__':
    print("\n" + "="*20 + " Running indexer.py (ChromaDB) Direct Execution Test " + "="*20)
    
    dummy_docs_with_embeddings = [
        {
            "id": "DUMMY_PROC1_chunk1",
            "text": "Este é o primeiro parágrafo do processo um.",
            "vector": [0.1] * EMBEDDING_MODEL_DIMENSION, 
            "processo": "PROC1-2024",
            "page_number": 1 
        },
        {
            "id": "DUMMY_PROC1_chunk2",
            "text": "Este é o segundo parágrafo do processo um.",
            "vector": [0.15] * EMBEDDING_MODEL_DIMENSION,
            "processo": "PROC1-2024",
            "page_number": 1
        },
        {
            "id": "DUMMY_PROC2_chunk1",
            "text": "Este é o único parágrafo do processo dois.",
            "vector": [0.2] * EMBEDDING_MODEL_DIMENSION,
            "processo": "PROC2-2023",
            "page_number": 5
        }
    ]

    print(f"CHROMA_PERSIST_DIR: {os.path.abspath(CHROMA_PERSIST_DIR)}")
    print(f"CHROMA_COLLECTION_NAME: {CHROMA_COLLECTION_NAME}")

    try:
        print(f"\nAttempting to create/update index with {len(dummy_docs_with_embeddings)} dummy documents...")
        create_or_update_index_chroma(dummy_docs_with_embeddings)
        print("\nChromaDB indexing test finished.")
        
        collection = _get_chroma_collection()
        print(f"Verification: Collection '{collection.name}' now contains {collection.count()} items.")

    except Exception as e:
        print(f"An error occurred during the indexer.py direct execution test: {e}", exc_info=True)

    print("\n" + "="*60 + "\n") 