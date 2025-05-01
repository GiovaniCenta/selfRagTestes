import logging
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any, Optional, Tuple
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - RERANKER - %(levelname)s - %(message)s')

# --- Constants ---
# You can choose base or large. Base is smaller and faster.
# RERANKER_MODEL_ID = "BAAI/bge-reranker-large"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"
RERANKER_SCORE_THRESHOLD = 0.1 # Default threshold for relevance after rerank
RERANKER_SELECT_TOP_N = 3      # Default number of top chunks to select after rerank

# --- Cached Model ---
_reranker_model_instance: Optional[CrossEncoder] = None

def load_reranker_model(model_id: str = RERANKER_MODEL_ID) -> CrossEncoder:
    """Loads and caches the CrossEncoder reranker model."""
    global _reranker_model_instance

    if _reranker_model_instance is not None:
        logging.debug("Returning cached reranker model.")
        return _reranker_model_instance

    logging.info(f"Loading reranker model: {model_id}")
    try:
        # Force reranker onto CPU to save VRAM for main LLM
        device = 'cpu'
        logging.info(f"Using device: {device} for reranker model")

        # max_length can be adjusted based on typical query+chunk length if needed
        _reranker_model_instance = CrossEncoder(model_id, max_length=512, device=device)
        logging.info("Reranker model loaded successfully.")
        return _reranker_model_instance
    except Exception as e:
        logging.error(f"Failed to load reranker model '{model_id}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load reranker model: {e}") from e

def rerank_chunks(
    query_claim: str,
    chunks: List[Dict[str, Any]],
    model_id: str = RERANKER_MODEL_ID
) -> List[Dict[str, Any]]:
    """Reranks a list of chunks based on their relevance to query claim using CrossEncoder.

    Args:
        query_claim: The query text (claim).
        chunks: A list of dictionaries, where each dictionary represents a chunk
                and must contain at least 'text' key. Other keys like 'metadata',
                'distance' from initial retrieval will be preserved.
        model_id: The Hugging Face ID of CrossEncoder model to use.

    Returns:
        A new list of input chunk dictionaries, sorted by reranker score
        in descending order (most relevant first). Each dictionary will have
        new key 'rerank_score' added.
        Returns empty list if input chunks are empty or reranking fails.
    """
    if not query_claim or not chunks:
        logging.warning("Received empty query or chunks for reranking. Returning empty list.")
        return []

    try:
        # Load reranker model (uses cache)
        model = load_reranker_model(model_id)

        # Prepare pairs for model
        chunk_texts = [chunk.get('text', '') for chunk in chunks]
        sentence_pairs = [[query_claim, chunk_text] for chunk_text in chunk_texts]

        # Compute scores
        logging.info(f"Computing rerank scores for {len(sentence_pairs)} pairs...")
        scores = model.predict(sentence_pairs, show_progress_bar=False)
        logging.info("Rerank scores computed.")

        # Add scores to chunks and handle potential errors
        if len(scores) != len(chunks):
             logging.error(f"Mismatch between number of scores ({len(scores)}) and chunks ({len(chunks)}). Aborting rerank.")
             # Optionally return original chunks or raise error
             return [] # Return empty to signal failure clearly

        chunks_with_scores = []
        for i, chunk in enumerate(chunks):
            # Create copy to avoid modifying original input list directly if passed around
            chunk_copy = chunk.copy()
            chunk_copy['rerank_score'] = float(scores[i]) # Ensure score is float
            chunks_with_scores.append(chunk_copy)

        # Sort chunks by score (descending)
        chunks_with_scores.sort(key=lambda x: x['rerank_score'], reverse=True)

        return chunks_with_scores

    except Exception as e:
        logging.error(f"An unexpected error occurred during reranking: {e}", exc_info=True)
        return [] # Return empty on error

# --- Example Usage Block (for direct testing) ---
if __name__ == "__main__":
    print("\n" + "="*20 + " Running reranker.py Direct Execution Test " + "="*20)

    # --- Dummy Data (Simulating retriever output) ---
    sample_query = "Qual o impacto da decisão no teto remuneratório?"
    sample_chunks_data = [
        {
            'text': "O tribunal decidiu que o teto se aplica imediatamente a todos os empregados.",
            'metadata': {'source': 'docA', 'page': 5},
            'distance': 0.2 # Example distance from initial retrieval
        },
        {
            'text': "A análise considerou os argumentos das partes sobre a natureza da empresa.",
            'metadata': {'source': 'docA', 'page': 4},
            'distance': 0.3
        },
        {
            'text': "O impacto financeiro da aplicação do teto foi discutido na seção V.",
            'metadata': {'source': 'docA', 'page': 12},
            'distance': 0.15
        },
        {
            'text': "Não houve menção explícita ao teto remuneratório neste parágrafo sobre contratos.",
            'metadata': {'source': 'docA', 'page': 8},
            'distance': 0.4
        }
    ]

    print(f'Test Query: "{sample_query}"')
    print(f"Input Chunks (Count: {len(sample_chunks_data)}):")
    for i, chunk in enumerate(sample_chunks_data):
         print(f"  Chunk {i+1}: '{chunk['text'][:50].replace(chr(10), ' ')}...' (Dist: {chunk.get('distance', 'N/A')})")
    print("-" * 20)

    try:
        # --- Run Reranking ---
        # The first time this runs, it will load the reranker model (can take time)
        reranked_results = rerank_chunks(sample_query, sample_chunks_data)

    except RuntimeError as e:
         print(f"RUNTIME ERROR during test: {e}")
         print("This might be an model loading failure. Check logs.")
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")
        reranked_results = [] # Ensure it's defined for printing


    # --- Print Result ---
    print("-" * 20)
    if reranked_results:
        print("--- Reranked Results (Sorted by Relevance) ---")
        for i, chunk in enumerate(reranked_results):
             print(f"Rank {i+1}: Score={chunk['rerank_score']:.4f}")
             print(f"  Text: '{chunk['text'][:80].replace(chr(10), ' ')}...'")
             print(f"  Original Distance: {chunk.get('distance', 'N/A')}")
             print(f"  Metadata: {chunk.get('metadata', {})}")
             print("--")
    elif sample_chunks_data: # Check if input was not empty
        print("--- Reranking FAILED ---")
        print("Check logs above for errors.")
    else:
        print("--- Reranking Skipped (No input) ---")


    print("\n" + "="*20 + " End of reranker.py Direct Execution Test " + "="*20) 