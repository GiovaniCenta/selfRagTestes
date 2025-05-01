import os
import sys
import argparse
import logging
import time
from typing import Tuple

# --- Configure logging ---
# Define logger level (e.g., INFO, DEBUG)
log_level = logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - MAIN - %(levelname)s - %(message)s')

# Add src directory to sys.path to ensure modules are found
# This assumes main.py is in the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

# --- Import project modules ---
try:
    from src.data_loader import load_and_prepare_data
    from src.retriever import retrieve_relevant_chunks
    from src.validator import validate_claim_with_llm
    from src.reranker import rerank_chunks
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}. Make sure src is in the Python path.")
    sys.exit(1)

# --- Constants for RAG pipeline ---
INITIAL_RETRIEVAL_TOP_K = 5 # Retrieve more candidates initially
RERANKER_SELECT_TOP_N = 2 # Select top N after reranking
RERANKER_SCORE_THRESHOLD = 0.1 # Optional: Minimum score from reranker to be considered relevant (adjust based on model/testing)

def run_validation_pipeline(acordao_path: str, resumo_path: str) -> Tuple[list, dict]:
    """Runs the core validation pipeline: load -> retrieve -> rerank -> validate.

    Args:
        acordao_path: Path to the main acórdão document.
        resumo_path: Path to the summary document.

    Returns:
        A tuple containing:
        - list: List of dictionaries, each with validation results for a claim.
        - dict: Dictionary containing collected metrics.
        Returns ([], {}) if the process fails early.
    """
    # 1. Load Data
    logging.info("Carregando e preparando dados...")
    acordao_chunks, resumo_claims = load_and_prepare_data(acordao_path, resumo_path)
    if acordao_chunks is None or resumo_claims is None:
        logging.error("Falha ao carregar dados.")
        return [], {} # Return empty results and metrics

    if not resumo_claims:
         logging.warning("Nenhuma alegação encontrada no resumo.")
         return [], {}

    # Initialize metrics collectors
    metrics = {
        "total_claims": len(resumo_claims),
        "processed_claims": 0,
        "retrieval_failures": 0,
        "llm_validation_failures": 0, # Includes None return or status "Erro"
        "llm_parsing_failures": 0,
        "total_latency_seconds": 0,
        "total_retrieval_latency": 0,
        "total_llm_latency": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_rerank_latency": 0,
        "claims_with_no_reranked_context": 0
    }

    # 3. Process Alegações
    validation_results = []
    logging.info(f"Iniciando validação de {metrics['total_claims']} alegações...")
    pipeline_start_time = time.time()

    for i, claim_data in enumerate(resumo_claims):
        claim_index = i + 1
        claim_text = claim_data.get('text', '')
        claim_processing_start_time = time.time()

        if not claim_text:
            logging.warning(f"Alegação {claim_index} está vazia. Pulando.")
            metrics["processed_claims"] += 1 # Count as processed but failed early
            metrics["llm_validation_failures"] += 1
            validation_results.append({
                "claim_index": claim_index,
                "claim_text": "[VAZIA]",
                "status": "Erro",
                "justification": "Alegação estava vazia no arquivo de resumo."
            })
            continue

        logging.info(f"--- Processando Alegação {claim_index}/{metrics['total_claims']} ---")
        logging.info(f"Alegação: {claim_text[:150].replace(chr(10), ' ')}...")

        # 4. Recuperar Contexto Inicial
        logging.info(f"Recuperando {INITIAL_RETRIEVAL_TOP_K} candidatos iniciais...")
        retrieval_start_time = time.time()
        retrieved_context = retrieve_relevant_chunks(claim_text, top_k=INITIAL_RETRIEVAL_TOP_K)
        retrieval_time = time.time() - retrieval_start_time
        metrics["total_retrieval_latency"] += retrieval_time
        logging.debug(f"Initial retrieval took {retrieval_time:.2f}s")

        status = "Erro"
        justification = "Falha desconhecida."
        input_tokens = 0
        output_tokens = 0
        llm_time = 0
        final_context_for_llm = []
        rerank_time = 0

        if retrieved_context is None or not retrieved_context.get('documents') or not retrieved_context['documents'][0]:
            logging.warning("Falha na recuperação inicial ou nenhum documento encontrado.")
            justification = "Falha na recuperação inicial de trechos."
            metrics["retrieval_failures"] += 1
            metrics["llm_validation_failures"] += 1
        else:
            initial_chunks_data = []
            # Reconstruct list of dicts for reranker input
            ids = retrieved_context.get('ids', [[]])[0]
            documents = retrieved_context.get('documents', [[]])[0]
            metadatas = retrieved_context.get('metadatas', [[]])[0]
            distances = retrieved_context.get('distances', [[]])[0]
            for idx in range(len(ids)):
                 initial_chunks_data.append({
                      'id': ids[idx],
                      'text': documents[idx],
                      'metadata': metadatas[idx],
                      'distance': distances[idx]
                 })

            # 4.5 Re-rank Chunks
            logging.info(f"Re-ranking {len(initial_chunks_data)} chunks...")
            rerank_start_time = time.time()
            reranked_chunks = rerank_chunks(claim_text, initial_chunks_data)
            rerank_time = time.time() - rerank_start_time
            metrics["total_rerank_latency"] += rerank_time
            logging.debug(f"Reranking took {rerank_time:.2f}s")

            if not reranked_chunks:
                logging.warning("Reranking falhou ou retornou lista vazia.")
                justification = "Falha durante o processo de re-ranking."
                metrics["llm_validation_failures"] += 1 # Treat as validation failure
            else:
                # Log reranked scores (optional)
                if log_level <= logging.DEBUG:
                    logging.debug("Reranked Chunks (Top 5):")
                    for r_idx, r_chunk in enumerate(reranked_chunks[:5]):
                         logging.debug(f"  - Rank {r_idx+1}: Score={r_chunk['rerank_score']:.4f} | Dist={r_chunk['distance']:.4f} | Text: {r_chunk['text'][:50]}...")

                # Select top N *after* reranking and apply score threshold
                selected_chunks = []
                for chunk in reranked_chunks[:RERANKER_SELECT_TOP_N]:
                    if chunk['rerank_score'] >= RERANKER_SCORE_THRESHOLD:
                        selected_chunks.append(chunk)
                    else:
                        logging.debug(f"Chunk descartado pelo threshold do reranker (Score: {chunk['rerank_score']:.4f} < {RERANKER_SCORE_THRESHOLD})")
                        # Optional: break here if you only want contiguous top chunks above threshold

                if not selected_chunks:
                    logging.warning(f"Nenhum chunk passou no threshold do reranker (Score >= {RERANKER_SCORE_THRESHOLD}).")
                    justification = f"Nenhum trecho considerado suficientemente relevante após re-ranking (limite={RERANKER_SCORE_THRESHOLD})."
                    metrics["claims_with_no_reranked_context"] += 1
                    metrics["llm_validation_failures"] += 1
                else:
                    logging.info(f"Selecionados {len(selected_chunks)} chunks após reranking e threshold.")
                    # Prepare context for LLM (just the text)
                    final_context_for_llm = [chunk['text'] for chunk in selected_chunks]
                    # Construct a simplified context dict for the validator function
                    # (We lost original distances/IDs structure here, but validator only needs docs)
                    context_for_validator = {'documents': [final_context_for_llm]}

                    # 5. Validar com LLM (using selected context)
                    logging.info("Validando alegação com LLM...")
                    llm_start_time = time.time()
                    result = validate_claim_with_llm(claim_text, context_for_validator)
                    llm_time = time.time() - llm_start_time
                    metrics["total_llm_latency"] += llm_time
                    logging.debug(f"LLM validation took {llm_time:.2f}s")

                    if result is None:
                        logging.error("Falha na validação do LLM (retornou None). Marcando como Erro.")
                        justification = "Falha interna durante a validação do LLM (ver logs do validator)."
                        metrics["llm_validation_failures"] += 1
                    else:
                        status = result.get("Resultado", "Erro")
                        justification = result.get("Justificativa", "Erro ao parsear justificativa.")
                        input_tokens = result.get("input_tokens", 0)
                        output_tokens = result.get("output_tokens", 0)
                        metrics["total_input_tokens"] += input_tokens
                        metrics["total_output_tokens"] += output_tokens

                        # Check for parsing failure specifically
                        if status == "Erro" and "Falha ao parsear" in justification or "Formato inesperado" in justification:
                             metrics["llm_parsing_failures"] += 1
                             metrics["llm_validation_failures"] += 1 # Also counts as validation failure
                        elif status == "Erro": # Other internal error during LLM step
                             metrics["llm_validation_failures"] += 1

        # Store results
        validation_results.append({
            "claim_index": claim_index,
            "claim_text": claim_text,
            "status": status,
            "justification": justification
        })
        metrics["processed_claims"] += 1
        logging.info(f"Resultado Alegação {claim_index}: {status}")
        claim_total_time = time.time() - claim_processing_start_time
        logging.info(f"Tempo para Alegação {claim_index}: {claim_total_time:.2f}s (Retrieval: {retrieval_time:.2f}s, Rerank: {rerank_time:.2f}s, LLM: {llm_time:.2f}s)")

    metrics["total_latency_seconds"] = time.time() - pipeline_start_time
    logging.info(f"Validação de todas as {metrics['processed_claims']}/{metrics['total_claims']} alegações concluída em {metrics['total_latency_seconds']:.2f} segundos.")
    return validation_results, metrics

def save_results(results: list, acordao_path: str, output_path: str):
    """Formats and saves the validation results to a text file."""
    logging.info(f"Salvando {len(results)} resultados em: {output_path}")
    try:
        # Extract process name from the acordao path
        process_name = os.path.basename(acordao_path).split('.')[0]
        # A more robust way might involve looking into metadata if available

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Processo {process_name}.\n") # Corrected: Added closing parenthesis
            for res in results:
                f.write(f"Reivindicação {res['claim_index']}: \"{res['claim_text']}\" → {res['status']}.\n") # Added trailing period and newline here too for clarity
                # Append justification only if Incorrect and justification is meaningful
                if res['status'] == 'Incorreta' and res['justification'] and res['justification'] != 'N/A':
                    # Requirement: "Trecho do acórdão: ..." but LLM gives justification.
                    # Using LLM justification here.
                    f.write(f" Justificativa: {res['justification']}")
                f.write("\n") # Newline after each claim
        logging.info("Resultados salvos com sucesso.")
    except Exception as e:
        logging.error(f"Falha ao salvar arquivo de saída: {e}", exc_info=True)

# --- New Function: Print Metrics ---
def print_metrics(metrics: dict):
    """Prints a formatted summary of the collected metrics."""
    logging.info("--- Métricas de Execução ---")
    print("\n--- Métricas de Execução ---")

    total_claims = metrics.get("total_claims", 0)
    processed_claims = metrics.get("processed_claims", 0)
    if processed_claims == 0:
        print("Nenhuma alegação foi processada.")
        logging.warning("Nenhuma alegação foi processada para cálculo de métricas.")
        return

    # Latency
    total_latency = metrics.get("total_latency_seconds", 0)
    avg_latency_per_claim = total_latency / processed_claims if processed_claims else 0
    avg_retrieval_latency = metrics.get("total_retrieval_latency", 0) / processed_claims if processed_claims else 0
    avg_rerank_latency = metrics.get("total_rerank_latency", 0) / processed_claims if processed_claims else 0
    avg_llm_latency = metrics.get("total_llm_latency", 0) / processed_claims if processed_claims else 0
    print(f"Latência Total: {total_latency:.2f} segundos")
    print(f"Latência Média por Alegação: {avg_latency_per_claim:.2f} segundos")
    print(f"  - Média Recuperação (Retriever): {avg_retrieval_latency:.2f} segundos")
    print(f"  - Média Re-ranking: {avg_rerank_latency:.2f} segundos")
    print(f"  - Média Validação (LLM): {avg_llm_latency:.2f} segundos")
    logging.info(f"Latência: Total={total_latency:.2f}s, Média/Alegação={avg_latency_per_claim:.2f}s (Retriever={avg_retrieval_latency:.2f}s, Rerank={avg_rerank_latency:.2f}s, LLM={avg_llm_latency:.2f}s)")

    # Tokens
    total_input_tokens = metrics.get("total_input_tokens", 0)
    total_output_tokens = metrics.get("total_output_tokens", 0)
    total_tokens = total_input_tokens + total_output_tokens
    avg_input_tokens = total_input_tokens / processed_claims if processed_claims else 0
    avg_output_tokens = total_output_tokens / processed_claims if processed_claims else 0
    print(f"\nUso de Tokens LLM Total: {total_tokens}")
    print(f"  - Entrada Total: {total_input_tokens}")
    print(f"  - Saída Total: {total_output_tokens}")
    print(f"Uso de Tokens LLM Médio por Alegação:")
    print(f"  - Média Entrada: {avg_input_tokens:.1f}")
    print(f"  - Média Saída: {avg_output_tokens:.1f}")
    logging.info(f"Tokens LLM: Total={total_tokens} (In={total_input_tokens}, Out={total_output_tokens}), Média/Alegação (In={avg_input_tokens:.1f}, Out={avg_output_tokens:.1f})")

    # Reliability / Failures
    retrieval_failures = metrics.get("retrieval_failures", 0)
    llm_validation_failures = metrics.get("llm_validation_failures", 0)
    llm_parsing_failures = metrics.get("llm_parsing_failures", 0)
    claims_no_reranked = metrics.get("claims_with_no_reranked_context", 0)
    success_rate = ((processed_claims - llm_validation_failures) / processed_claims * 100) if processed_claims else 0
    print(f"\nConfiabilidade (em {processed_claims} alegações processadas):")
    print(f"  - Taxa de Sucesso (Validação Completa): {success_rate:.1f}%" )
    print(f"  - Falhas de Recuperação Inicial: {retrieval_failures}")
    print(f"  - Alegações Sem Contexto Pós-Rerank/Threshold: {claims_no_reranked}")
    print(f"  - Falhas de Validação (LLM Total): {llm_validation_failures}")
    print(f"    - Falhas de Parsing Resposta LLM: {llm_parsing_failures}")
    logging.info(f"Confiabilidade: Sucesso={success_rate:.1f}%, Falhas Retriever={retrieval_failures}, Sem Ctx Pós-Rerank={claims_no_reranked}, Falhas LLM={llm_validation_failures} (Parsing={llm_parsing_failures})")
    print("----------------------------")

# --- New Function: Print Results Summary ---
def print_results_summary(results: list):
    """Prints a formatted summary of the validation results to the console."""
    print("\n--- Resumo da Validação ---")
    if not results:
        print("Nenhum resultado de validação para exibir.")
        logging.warning("Nenhum resultado de validação disponível para exibir no console.")
        return

    for res in results:
        claim_text_short = res.get('claim_text', '[VAZIA]')
        # Limit text length for cleaner console output
        if len(claim_text_short) > 100:
            claim_text_short = claim_text_short[:100] + "..."
        status = res.get('status', 'Erro')
        justification = res.get('justification', '')

        print(f"  - Alegação {res.get('claim_index', '?')}: \"{claim_text_short}\" -> {status}")
        # Print justification only if status is 'Incorreta' and justification is meaningful
        if status == 'Incorreta' and justification and justification != 'N/A' and "Erro" not in justification:
             print(f"    Justificativa: {justification}")

    print("---------------------------\n")

def main():
    """Main function to parse arguments and run the validation pipeline."""
    parser = argparse.ArgumentParser(description="Valida alegações de um resumo contra um acórdão do TCU.")
    parser.add_argument("--acordao_file",
                        default=os.path.join("data", "Acórdão 733 de 2025 Plenário.pdf"),
                        help="Caminho para o arquivo do acórdão (default: data/Acórdão 733 de 2025 Plenário.pdf)")
    parser.add_argument("--resumo_file",
                        default=os.path.join("data", "Acórdão 733-2025 resumos.txt"),
                        help="Caminho para o arquivo de resumo (default: data/Acórdão 733-2025 resumos.txt)")
    parser.add_argument("-o", "--output_file",
                        default="validation_results.txt",
                        help="Caminho para salvar os resultados (default: validation_results.txt)")
    # parser.add_argument("--reindex", action="store_true", help="Força a recriação do índice ChromaDB.") # Example for later

    args = parser.parse_args()

    logging.info("--- Iniciando Processo de Validação ---")
    logging.info(f"Acórdão: {args.acordao_file}")
    logging.info(f"Resumo: {args.resumo_file}")
    logging.info(f"Arquivo de Saída: {args.output_file}")

    # Validate input file paths
    if not os.path.exists(args.acordao_file):
        logging.error(f"Erro: Arquivo do acórdão não encontrado em '{args.acordao_file}'")
        sys.exit(1)
    if not os.path.exists(args.resumo_file):
        logging.error(f"Erro: Arquivo do resumo não encontrado em '{args.resumo_file}'")
        sys.exit(1)

    # Run the main processing pipeline
    start_time_main = time.time()
    # Capture both results and metrics
    results, metrics = run_validation_pipeline(args.acordao_file, args.resumo_file)
    pipeline_time = time.time() - start_time_main
    logging.info(f"Pipeline de validação executado em {pipeline_time:.2f}s")

    # Save results if successful
    if results:
        save_results(results, args.acordao_file, args.output_file)
    else:
        logging.error("Pipeline de validação não retornou resultados. Arquivo de saída não será gerado.")

    # Print metrics summary
    if metrics:
        print_metrics(metrics)
    else:
        logging.warning("Nenhuma métrica foi coletada.")

    # Print results summary to console
    if results:
        print_results_summary(results)
    else:
        logging.info("Nenhum resultado para exibir no resumo do console.")

    logging.info("--- Processo de Validação Concluído ---")

if __name__ == "__main__":
    main() 