import logging
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - RERANKER - %(levelname)s - %(message)s')

# --- Constants --- 
CROSS_ENCODER_MODEL_NAME = "rufimelo/Legal-BERTimbau-sts-large"

# --- Cached Model Instance --- 
_cross_encoder_instance: Optional[CrossEncoder] = None

def get_cross_encoder_model(model_name: str = CROSS_ENCODER_MODEL_NAME) -> CrossEncoder:
    """Loads or returns the cached CrossEncoder model."""
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        logging.info(f"Loading CrossEncoder model: {model_name}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # max_length for CrossEncoder is important for context handling
        _cross_encoder_instance = CrossEncoder(model_name, max_length=512, device=device)
        logging.info(f"CrossEncoder model loaded on device: {device}")
    return _cross_encoder_instance

def rank(query: str, docs: List[Dict], n: int = 5) -> List[Dict]:
    """
    Reranks a list of documents based on their relevance to a query using a CrossEncoder model.
    Expects documents to have a 'text' field.

    Args:
        query: The query string.
        docs: A list of document dictionaries, each containing at least a 'text' field.
        n: The number of top documents to return.

    Returns:
        A list of the top N documents, sorted by reranked score in descending order.
        Each document dictionary will have an added 'reranker_score' field.
        Returns an empty list if an error occurs or no documents are provided.
    """
    if not query or not docs:
        logging.warning("Query and documents list must be provided for reranking.")
        return []

    try:
        cross_encoder = get_cross_encoder_model()
        
        # Prepare pairs for the CrossEncoder: (query, doc_text)
        sentence_pairs = [(query, doc.get('text', '')) for doc in docs]
        
        if not sentence_pairs:
            logging.warning("No text found in provided documents for reranking.")
            return []

        logging.info(f"Reranking {len(sentence_pairs)} documents with CrossEncoder...")
        # The CrossEncoder predict method returns scores directly
        scores = cross_encoder.predict(sentence_pairs, show_progress_bar=True)
        logging.info("Reranking complete.")

        # Add scores to documents and sort
        for doc, score in zip(docs, scores):
            doc['reranker_score'] = float(score) # Ensure score is float
        
        # Sort documents by reranker_score in descending order
        ranked_docs = sorted(docs, key=lambda x: x.get('reranker_score', 0.0), reverse=True)
        
        top_n_docs = ranked_docs[:n]
        logging.info(f"Returning top {len(top_n_docs)} reranked documents.")
        
        return top_n_docs

    except Exception as e:
        logging.error(f"Failed to rerank documents for query '{query[:50]}...': {e}", exc_info=True)
        return []

# Example (for testing if run directly)
if __name__ == '__main__':
    test_query = "princípio da insignificância em crimes de furto"
    test_docs_from_retriever = [
        {"id": "doc1", "processo": "111-2024", "text": "O princípio da insignificância aplica-se a crimes de furto quando o valor do bem é ínfimo.", "SimilarityScore": 0.8},
        {"id": "doc2", "processo": "111-2024", "text": "A jurisprudência do STF sobre o princípio da insignificância é vasta.", "SimilarityScore": 0.75},
        {"id": "doc3", "processo": "111-2024", "text": "Contratos administrativos e suas cláusulas exorbitantes.", "SimilarityScore": 0.3},
        {"id": "doc4", "processo": "111-2024", "text": "O furto qualificado não admite, em regra, o princípio da insignificância.", "SimilarityScore": 0.85},
        {"id": "doc5", "processo": "111-2024", "text": "Para a aplicação do princípio da insignificância, analisa-se o desvalor da conduta e do resultado.", "SimilarityScore": 0.78},
        {"id": "doc6", "processo": "111-2024", "text": "Recurso especial sobre matéria constitucional não é cabível.", "SimilarityScore": 0.2}
    ]

    print(f"Testing reranker with query: '{test_query}'")
    print(f"Original number of docs: {len(test_docs_from_retriever)}")

    top_reranked_docs = rank(test_query, test_docs_from_retriever, n=3)

    if top_reranked_docs:
        print(f"\nTop {len(top_reranked_docs)} reranked documents:")
        for i, doc in enumerate(top_reranked_docs):
            print(f"  Rank {i+1}: ID={doc['id']}, Reranker Score={doc['reranker_score']:.4f}")
            print(f"    Original Score: {doc.get('SimilarityScore', 'N/A')}")
            print(f"    Text: {doc['text'][:100]}...")
    else:
        print("\nFailed to rerank documents or no documents returned.") 