import os
import sys
import argparse
import logging
import time
from typing import Tuple, List, Dict, Any
import csv

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
    from src.indexer import get_embedding_model, create_or_update_index, CHROMA_PERSIST_DIR
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}. Make sure src is in the Python path.")
    sys.exit(1)

# --- Constants for RAG pipeline ---
INITIAL_RETRIEVAL_TOP_K = 5 # Retrieve more candidates initially
RERANKER_SELECT_TOP_N = 5 # Select top N after reranking
RERANKER_SCORE_THRESHOLD = -10 # Optional: Minimum score from reranker to be considered relevant (adjust based on model/testing)

def run_validation_pipeline(acordao_path: str, resumo_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs the core validation pipeline: load -> retrieve -> rerank -> validate.

    Args:
        acordao_path: Path to the main acórdão document.
        resumo_path: Path to the summary document.

    Returns:
        A tuple containing:
        - list: List of dictionaries, each with validation results for a claim (sentence).
                Includes 'input_tokens', 'output_tokens', 'claim_processing_time', 
                'top_rerank_score', and 'used_context_for_llm'.
        - dict: Dictionary containing collected metrics.
        Returns ([], {}) if the process fails early.
    """
    # 1. Load Data
    logging.info("Carregando e preparando dados...")
    # data_loader now splits resumo into sentences, each is a "claim"
    acordao_chunks, resumo_claims = load_and_prepare_data(acordao_path, resumo_path)
    if acordao_chunks is None or resumo_claims is None:
        logging.error("Falha ao carregar dados.")
        return [], {}

    if not resumo_claims:
         logging.warning("Nenhuma alegação (sentença) encontrada no resumo.")
         return [], {}

    # Initialize metrics collectors
    metrics = {
        "total_claims": len(resumo_claims),
        "processed_claims": 0,
        "retrieval_failures": 0,
        "llm_validation_failures": 0,
        "llm_parsing_failures": 0,
        "total_latency_seconds": 0,
        "total_retrieval_latency": 0,
        "total_llm_latency": 0,
        "total_input_tokens": 0, # Aggregated over all claims
        "total_output_tokens": 0, # Aggregated over all claims
        "total_rerank_latency": 0,
        "claims_with_no_reranked_context": 0
    }

    # Process Alegações (Sentences)
    validation_results = []
    logging.info(f"Iniciando validação de {metrics['total_claims']} alegações (sentenças)...")
    pipeline_start_time = time.time()

    for i, claim_data in enumerate(resumo_claims):
        claim_index = i + 1
        claim_text = claim_data.get('text', '') # claim_text is now a sentence
        claim_processing_start_time = time.time()

        # Initialize per-claim specific metrics/info
        claim_input_tokens = 0
        claim_output_tokens = 0
        claim_top_rerank_score = None # For 'Confianca' proxy
        retrieval_time = 0 # Initialize for this claim's scope
        rerank_time = 0    # Initialize for this claim's scope
        llm_time = 0       # Initialize for this claim's scope
        used_context_for_llm = [] # Initialize to store the context used for this claim

        if not claim_text:
            logging.warning(f"Alegação (sentença) {claim_index} está vazia. Pulando.")
            metrics["processed_claims"] += 1
            metrics["llm_validation_failures"] += 1
            status = "Erro"
            justification = "Alegação (sentença) estava vazia no arquivo de resumo."
        else:
            logging.info(f"--- Processando Alegação (Sentença) {claim_index}/{metrics['total_claims']} ---")
            logging.info(f"Alegação: {claim_text[:150].replace(chr(10), ' ')}...")

            # 4. Recuperar Contexto Inicial
            logging.info(f"Recuperando {INITIAL_RETRIEVAL_TOP_K} candidatos iniciais...")
            retrieval_start_time = time.time()
            retrieved_context = retrieve_relevant_chunks(claim_text, top_k=INITIAL_RETRIEVAL_TOP_K)
            retrieval_time = time.time() - retrieval_start_time
            metrics["total_retrieval_latency"] += retrieval_time
            logging.debug(f"Initial retrieval took {retrieval_time:.2f}s")

            status = "Erro" # Default status
            justification = "Falha desconhecida." # Default justification

            if retrieved_context is None or not retrieved_context.get('documents') or not retrieved_context['documents'][0]:
                logging.warning("Falha na recuperação inicial ou nenhum documento encontrado.")
                justification = "Falha na recuperação inicial de trechos."
                metrics["retrieval_failures"] += 1
                metrics["llm_validation_failures"] += 1
            else:
                initial_chunks_data = []
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
                reranked_chunks_list = rerank_chunks(claim_text, initial_chunks_data)
                rerank_time = time.time() - rerank_start_time
                metrics["total_rerank_latency"] += rerank_time
                logging.debug(f"Reranking took {rerank_time:.2f}s")

                if not reranked_chunks_list:
                    logging.warning("Reranking falhou ou retornou lista vazia.")
                    justification = "Falha durante o processo de re-ranking."
                    metrics["llm_validation_failures"] += 1
                else:
                    if log_level <= logging.DEBUG:
                        logging.debug("Reranked Chunks (Top 5):")
                        for r_idx, r_chunk in enumerate(reranked_chunks_list[:5]):
                             logging.debug(f"  - Rank {r_idx+1}: Score={r_chunk['rerank_score']:.4f} | Dist={r_chunk['distance']:.4f} | Text: {r_chunk['text'][:50]}...")

                    selected_chunks_for_llm_data = []
                    for chunk in reranked_chunks_list[:RERANKER_SELECT_TOP_N]:
                        if chunk['rerank_score'] >= RERANKER_SCORE_THRESHOLD:
                            selected_chunks_for_llm_data.append(chunk)
                        else:
                            logging.debug(f"Chunk descartado pelo threshold do reranker (Score: {chunk['rerank_score']:.4f} < {RERANKER_SCORE_THRESHOLD})")

                    if not selected_chunks_for_llm_data:
                        logging.warning(f"Nenhum chunk passou no threshold do reranker (Score >= {RERANKER_SCORE_THRESHOLD}).")
                        justification = f"Nenhum trecho considerado suficientemente relevante após re-ranking (limite={RERANKER_SCORE_THRESHOLD})."
                        metrics["claims_with_no_reranked_context"] += 1
                        metrics["llm_validation_failures"] += 1
                    else:
                        logging.info(f"Selecionados {len(selected_chunks_for_llm_data)} chunks após reranking e threshold.")
                        if selected_chunks_for_llm_data:
                            claim_top_rerank_score = selected_chunks_for_llm_data[0]['rerank_score']

                        used_context_for_llm = [chunk['text'] for chunk in selected_chunks_for_llm_data]
                        context_for_validator = {'documents': [used_context_for_llm]}

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
                            claim_input_tokens = result.get("input_tokens", 0)
                            claim_output_tokens = result.get("output_tokens", 0)
                            metrics["total_input_tokens"] += claim_input_tokens
                            metrics["total_output_tokens"] += claim_output_tokens

                            if status == "Erro" and ("Falha ao parsear" in justification or "Formato inesperado" in justification):
                                 metrics["llm_parsing_failures"] += 1
                                 metrics["llm_validation_failures"] += 1
                            elif status == "Erro":
                                 metrics["llm_validation_failures"] += 1
        
        claim_processing_time = time.time() - claim_processing_start_time
        validation_results.append({
            "claim_index": claim_index,
            "claim_text": claim_text,
            "status": status,
            "justification": justification,
            "input_tokens": claim_input_tokens,
            "output_tokens": claim_output_tokens,
            "claim_processing_time": claim_processing_time,
            "top_rerank_score": claim_top_rerank_score,
            "used_context_for_llm": used_context_for_llm
        })
        metrics["processed_claims"] += 1
        logging.info(f"Resultado Alegação (Sentença) {claim_index}: {status}")
        logging.info(f"Tempo para Alegação (Sentença) {claim_index}: {claim_processing_time:.2f}s (Rec: {retrieval_time:.2f}s, Rerank: {rerank_time:.2f}s, LLM: {llm_time:.2f}s)")

    metrics["total_latency_seconds"] = time.time() - pipeline_start_time
    logging.info(f"Validação de todas as {metrics['processed_claims']}/{metrics['total_claims']} alegações (sentenças) concluída em {metrics['total_latency_seconds']:.2f} segundos.")
    return validation_results, metrics

def save_results(results: list, acordao_path: str, output_path: str):
    """Formats and saves the validation results to a text file."""
    logging.info(f"Salvando {len(results)} resultados em formato TXT: {output_path}")
    try:
        process_name = os.path.basename(acordao_path).split('.')[0]
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Processo {process_name}.\n")
            for res in results:
                f.write(f"Reivindicação {res['claim_index']} (Sentença): \"{res['claim_text']}\" → {res['status']}.\n")
                if res['status'] == 'Incorreta' and res['justification'] and res['justification'] != 'N/A':
                    f.write(f" Justificativa: {res['justification']}\n") # Added newline for justification as well
                f.write("\n") # Extra newline between claims for better readability
        logging.info("Resultados TXT salvos com sucesso.")
    except Exception as e:
        logging.error(f"Falha ao salvar arquivo de saída TXT: {e}", exc_info=True)

# --- New Function: Save Results to CSV ---
def save_results_csv(results: List[Dict[str, Any]], output_filename: str):
    """Saves the detailed validation results to a CSV file."""
    if not results:
        logging.warning("Nenhum resultado para salvar em CSV.")
        return

    logging.info(f"Salvando {len(results)} resultados detalhados em CSV: {output_filename}")
    fieldnames = ['ID', 'Sentenca', 'Prediction', 'Confianca', 'numero_tokens', 'runtime_seconds']

    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for item in results:
                num_tokens = item.get('input_tokens', 0) + item.get('output_tokens', 0)
                confidence_score = item.get('top_rerank_score')
                
                writer.writerow({
                    'ID': item.get('claim_index', ''),
                    'Sentenca': item.get('claim_text', ''),
                    'Prediction': item.get('status', 'Erro'),
                    'Confianca': f"{confidence_score:.4f}" if confidence_score is not None else "N/A",
                    'numero_tokens': num_tokens,
                    'runtime_seconds': f"{item.get('claim_processing_time', 0):.2f}"
                })
        logging.info(f"Resultados CSV salvos com sucesso em {output_filename}")
    except Exception as e:
        logging.error(f"Falha ao salvar arquivo CSV: {e}", exc_info=True)

# --- New Function: Save Justifications and Context to CSV ---
def save_justifications_csv(results: List[Dict[str, Any]], output_filename: str):
    """Saves the ID, Sentence, Prediction, Justification, and Used Context to a CSV file."""
    if not results:
        logging.warning("Nenhum resultado para salvar justificativas em CSV.")
        return

    logging.info(f"Salvando {len(results)} justificativas e contextos em CSV: {output_filename}")
    fieldnames = ['ID', 'Sentenca', 'Prediction', 'Justificativa', 'ContextoUtilizado']

    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for item in results:
                context_str = " [SEP_CONTEXTO] ".join(item.get('used_context_for_llm', []))
                writer.writerow({
                    'ID': item.get('claim_index', ''),
                    'Sentenca': item.get('claim_text', ''),
                    'Prediction': item.get('status', 'Erro'),
                    'Justificativa': item.get('justification', 'N/A'),
                    'ContextoUtilizado': context_str
                })
        logging.info(f"Justificativas e contextos CSV salvos com sucesso em {output_filename}")
    except Exception as e:
        logging.error(f"Falha ao salvar arquivo CSV de justificativas e contextos: {e}", exc_info=True)

# --- New Function: Print Metrics ---
def print_metrics(metrics: dict):
    """Prints a formatted summary of the collected metrics."""
    logging.info("--- Métricas de Execução ---")
    print("\n--- Métricas de Execução ---")

    total_claims = metrics.get("total_claims", 0)
    processed_claims = metrics.get("processed_claims", 0)
    if processed_claims == 0:
        print("Nenhuma alegação (sentença) foi processada.")
        logging.warning("Nenhuma alegação (sentença) foi processada para cálculo de métricas.")
        return

    # Latency
    total_latency = metrics.get("total_latency_seconds", 0)
    avg_latency_per_claim = total_latency / processed_claims if processed_claims else 0
    avg_retrieval_latency = metrics.get("total_retrieval_latency", 0) / processed_claims if processed_claims else 0
    avg_rerank_latency = metrics.get("total_rerank_latency", 0) / processed_claims if processed_claims else 0
    avg_llm_latency = metrics.get("total_llm_latency", 0) / processed_claims if processed_claims else 0
    print(f"Latência Total: {total_latency:.2f} segundos")
    print(f"Latência Média por Alegação (Sentença): {avg_latency_per_claim:.2f} segundos")
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
    print(f"Uso de Tokens LLM Médio por Alegação (Sentença):")
    print(f"  - Média Entrada: {avg_input_tokens:.1f}")
    print(f"  - Média Saída: {avg_output_tokens:.1f}")
    logging.info(f"Tokens LLM: Total={total_tokens} (In={total_input_tokens}, Out={total_output_tokens}), Média/Alegação (In={avg_input_tokens:.1f}, Out={avg_output_tokens:.1f})")

    # Reliability / Failures
    retrieval_failures = metrics.get("retrieval_failures", 0)
    llm_validation_failures = metrics.get("llm_validation_failures", 0)
    llm_parsing_failures = metrics.get("llm_parsing_failures", 0)
    claims_no_reranked = metrics.get("claims_with_no_reranked_context", 0)
    success_rate = ((processed_claims - llm_validation_failures) / processed_claims * 100) if processed_claims else 0
    print(f"\nConfiabilidade (em {processed_claims} alegações/sentenças processadas):")
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
    print("\n--- Resumo da Validação (Sentenças) ---")
    if not results:
        print("Nenhum resultado de validação para exibir.")
        logging.warning("Nenhum resultado de validação disponível para exibir no console.")
        return

    for res in results:
        claim_text_short = res.get('claim_text', '[VAZIA]')
        # Limit text length for cleaner console output if needed, though sentences might be shorter
        if len(claim_text_short) > 100: # Keep this, some sentences can be long
            claim_text_short = claim_text_short[:100] + "..."
        status = res.get('status', 'Erro')
        justification = res.get('justification', '')

        print(f"  - Alegação (Sentença) {res.get('claim_index', '?')}: \"{claim_text_short}\" -> {status}")
        if status == 'Incorreta' and justification and justification != 'N/A' and "Erro" not in justification:
             print(f"    Justificativa: {justification}")
    print("---------------------------\n")

def main():
    """Main function to parse arguments and run the validation pipeline."""
    parser = argparse.ArgumentParser(description="Valida alegações (sentenças) de um resumo contra um acórdão do TCU.")
    parser.add_argument("--acordao_file",
                        default=os.path.join("data", "Acórdão 764 de 2025 Plenário.pdf"),
                        help="Caminho para o arquivo do acórdão (PDF) (default: data/Acórdão 764 de 2025 Plenário.pdf)")
    parser.add_argument("--resumo_file",
                        default=os.path.join("data", "Acórdão 764-2025 resumos.txt"),
                        help="Caminho para o arquivo de resumo (TXT) (default: data/Acórdão 764-2025 resumos.txt)")
    parser.add_argument("-o", "--output_file",
                        default="validation_results.txt",
                        help="Caminho para salvar os resultados em TXT (default: validation_results.txt)")
    parser.add_argument("--csv_output_file",
                        default="results/Acordao 764-2025 resultados.csv",
                        help="Caminho para salvar os resultados detalhados em CSV (default: results/Acordao 764-2025 resultados.csv)")
    parser.add_argument("--justificativas_file",
                        default="results/Acordao 764-2025 justificativas.csv",
                        help="Caminho para salvar as justificativas em CSV (default: results/Acordao 764-2025 justificativas.csv)")
    # Optional: Add a flag to force re-indexing if desired
    # parser.add_argument("--reindex", action="store_true", help="Força a recriação do índice ChromaDB.")

    args = parser.parse_args()

    # Ensure the results directory exists for CSV files
    # (Code to create output directories for results and justifications is assumed to be here or handled by save functions)
    # For example:
    for out_path in [args.csv_output_file, args.justificativas_file]:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            logging.info(f"Diretório de resultados criado: {out_dir}")

    logging.info("--- Iniciando Processo de Validação ---")
    logging.info(f"Acórdão: {args.acordao_file}")
    logging.info(f"Resumo: {args.resumo_file}")
    logging.info(f"Arquivo de Saída TXT: {args.output_file}")
    logging.info(f"Arquivo de Saída CSV (Resultados): {args.csv_output_file}")
    logging.info(f"Arquivo de Saída CSV (Justificativas): {args.justificativas_file}")

    # --- Validating input file paths before attempting indexing or validation --- 
    if not os.path.exists(args.acordao_file):
        logging.error(f"Erro: Arquivo do acórdão não encontrado em '{args.acordao_file}'. Não é possível criar índice ou validar.")
        sys.exit(1)
    # Resumo file is checked later, as indexing only needs acordao_file

    # --- Check and Create ChromaDB Index if it doesn't exist --- 
    # (or if --reindex flag is set, if you implement that)
    if not os.path.exists(CHROMA_PERSIST_DIR): # Or add: or args.reindex
        logging.info(f"Diretório do índice ChromaDB não encontrado em '{CHROMA_PERSIST_DIR}'.")
        logging.info(f"Tentando criar o índice a partir de: {args.acordao_file}")
        
        # 1. Load acórdão data specifically for indexing
        # Pass an empty string or None for resumo_path to signal only acórdão processing for index
        logging.info(f"Carregando acórdão '{args.acordao_file}' para indexação...")
        acordao_chunks_for_indexing, _ = load_and_prepare_data(args.acordao_file, "") 
        
        if acordao_chunks_for_indexing:
            logging.info(f"Preparando para indexar {len(acordao_chunks_for_indexing)} chunks do acórdão...")
            try:
                # 2. Load embedding model (indexer's own instance, or pass one if pre-loaded)
                embedding_model_for_indexing = get_embedding_model()
                # 3. Create/Update index
                create_or_update_index(acordao_chunks_for_indexing, embedding_model_for_indexing)
                logging.info(f"Índice ChromaDB criado/atualizado com sucesso em '{CHROMA_PERSIST_DIR}'.")
            except Exception as e:
                logging.error(f"Falha ao criar o índice ChromaDB: {e}", exc_info=True)
                logging.error("Prosseguindo sem índice. A validação provavelmente falhará na etapa de recuperação.")
                # Optionally: sys.exit(1) if indexing is critical and failed
        else:
            logging.warning(f"Nenhum chunk extraído de '{args.acordao_file}' para indexação. O índice não será criado.")
            # Let it proceed; retrieval will fail later if index is truly needed and not just empty.
    else:
        logging.info(f"Diretório do índice ChromaDB encontrado em '{CHROMA_PERSIST_DIR}'.")

    # --- Proceed with validation --- 
    # Validate resumo_file path now before running the main pipeline
    if not os.path.exists(args.resumo_file):
        logging.error(f"Erro: Arquivo do resumo não encontrado em '{args.resumo_file}'.")
        sys.exit(1)

    start_time_main = time.time()
    # Pass the full function definition for run_validation_pipeline here from your existing code
    # For this edit, I am replacing it with a placeholder call that would use your existing function
    # results, metrics = run_validation_pipeline(args.acordao_file, args.resumo_file)
    # Placeholder for actual call to the full run_validation_pipeline:
    # Replace this with the actual call to your full run_validation_pipeline function.
    # Ensure run_validation_pipeline is defined as before.
    # For the purpose of this edit, imagine the full run_validation_pipeline is here.
    # This section is just to show where the main logic sits after indexing.

    # --- Re-inserting the actual run_validation_pipeline call and subsequent logic --- 
    # The placeholder pass for run_validation_pipeline and subsequent functions needs to be 
    # replaced by their actual definitions from your working main.py. 
    # This edit focuses on adding the indexing logic within main().
    
    # Assuming run_validation_pipeline, save_results, save_results_csv, 
    # save_justifications_csv, print_metrics, print_results_summary are defined as before.
    # The following is the logical flow after the indexing part.

    # Actual pipeline execution
    logging.info("Iniciando pipeline de validação principal...")
    results, metrics = run_validation_pipeline(args.acordao_file, args.resumo_file)
    pipeline_time = time.time() - start_time_main
    logging.info(f"Pipeline de validação executado em {pipeline_time:.2f}s")

    if results:
        save_results(results, args.acordao_file, args.output_file)
        save_results_csv(results, args.csv_output_file)
        save_justifications_csv(results, args.justificativas_file)
    else:
        logging.error("Pipeline de validação não retornou resultados. Arquivos de saída não serão gerados.")

    if metrics:
        print_metrics(metrics)
    else:
        logging.warning("Nenhuma métrica foi coletada.")

    if results:
        print_results_summary(results)
    else:
        logging.info("Nenhum resultado para exibir no resumo do console.")

    logging.info("--- Processo de Validação Concluído ---")

if __name__ == "__main__":
    # For brevity, the definitions of run_validation_pipeline, save_results, 
    # save_results_csv, save_justifications_csv, print_metrics, print_results_summary
    # are assumed to be present above this main block, as they were in your original file.
    # The main change is the addition of the indexing logic within the main() function itself.
    def run_validation_pipeline(acordao_path: str, resumo_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # This is the full definition of run_validation_pipeline from the previous correct version.
        # It's re-inserted here to make the file complete and runnable.
        logging.info("Carregando e preparando dados...")
        acordao_chunks, resumo_claims = load_and_prepare_data(acordao_path, resumo_path)
        if acordao_chunks is None or resumo_claims is None:
            logging.error("Falha ao carregar dados.")
            return [], {}
        if not resumo_claims:
            logging.warning("Nenhuma alegação (sentença) encontrada no resumo.")
            return [], {}
        metrics = {
            "total_claims": len(resumo_claims), "processed_claims": 0, "retrieval_failures": 0,
            "llm_validation_failures": 0, "llm_parsing_failures": 0, "total_latency_seconds": 0,
            "total_retrieval_latency": 0, "total_llm_latency": 0, "total_input_tokens": 0,
            "total_output_tokens": 0, "total_rerank_latency": 0, "claims_with_no_reranked_context": 0
        }
        validation_results = []
        logging.info(f"Iniciando validação de {metrics['total_claims']} alegações (sentenças)...")
        pipeline_start_time = time.time()
        for i, claim_data in enumerate(resumo_claims):
            claim_index = i + 1; claim_text = claim_data.get('text', ''); claim_processing_start_time = time.time()
            claim_input_tokens = 0; claim_output_tokens = 0; claim_top_rerank_score = None
            retrieval_time = 0; rerank_time = 0; llm_time = 0; used_context_for_llm = []
            if not claim_text:
                logging.warning(f"Alegação (sentença) {claim_index} está vazia. Pulando.")
                metrics["processed_claims"] += 1; metrics["llm_validation_failures"] += 1
                status = "Erro"; justification = "Alegação (sentença) estava vazia no arquivo de resumo."
            else:
                logging.info(f"--- Processando Alegação (Sentença) {claim_index}/{metrics['total_claims']} ---")
                logging.info(f"Alegação: {claim_text[:150].replace(chr(10), ' ')}...")
                logging.info(f"Recuperando {INITIAL_RETRIEVAL_TOP_K} candidatos iniciais...")
                retrieval_start_time = time.time()
                retrieved_context = retrieve_relevant_chunks(claim_text, top_k=INITIAL_RETRIEVAL_TOP_K)
                retrieval_time = time.time() - retrieval_start_time
                metrics["total_retrieval_latency"] += retrieval_time
                logging.debug(f"Initial retrieval took {retrieval_time:.2f}s")
                status = "Erro"; justification = "Falha desconhecida."
                if retrieved_context is None or not retrieved_context.get('documents') or not retrieved_context['documents'][0]:
                    logging.warning("Falha na recuperação inicial ou nenhum documento encontrado.")
                    justification = "Falha na recuperação inicial de trechos."
                    metrics["retrieval_failures"] += 1; metrics["llm_validation_failures"] += 1
                else:
                    initial_chunks_data = []
                    ids = retrieved_context.get('ids', [[]])[0]; documents = retrieved_context.get('documents', [[]])[0]
                    metadatas = retrieved_context.get('metadatas', [[]])[0]; distances = retrieved_context.get('distances', [[]])[0]
                    for idx in range(len(ids)):
                        initial_chunks_data.append({'id': ids[idx], 'text': documents[idx], 'metadata': metadatas[idx], 'distance': distances[idx]})
                    logging.info(f"Re-ranking {len(initial_chunks_data)} chunks...")
                    rerank_start_time = time.time()
                    reranked_chunks_list = rerank_chunks(claim_text, initial_chunks_data)
                    rerank_time = time.time() - rerank_start_time
                    metrics["total_rerank_latency"] += rerank_time; logging.debug(f"Reranking took {rerank_time:.2f}s")
                    if not reranked_chunks_list:
                        logging.warning("Reranking falhou ou retornou lista vazia."); justification = "Falha durante o processo de re-ranking."
                        metrics["llm_validation_failures"] += 1
                    else:
                        if log_level <= logging.DEBUG:
                            logging.debug("Reranked Chunks (Top 5):")
                            for r_idx, r_chunk in enumerate(reranked_chunks_list[:5]):
                                logging.debug(f"  - Rank {r_idx+1}: Score={r_chunk['rerank_score']:.4f} | Dist={r_chunk['distance']:.4f} | Text: {r_chunk['text'][:50]}...")
                        selected_chunks_for_llm_data = []
                        for chunk in reranked_chunks_list[:RERANKER_SELECT_TOP_N]:
                            if chunk['rerank_score'] >= RERANKER_SCORE_THRESHOLD: selected_chunks_for_llm_data.append(chunk)
                            else: logging.debug(f"Chunk descartado pelo threshold do reranker (Score: {chunk['rerank_score']:.4f} < {RERANKER_SCORE_THRESHOLD})")
                        if not selected_chunks_for_llm_data:
                            logging.warning(f"Nenhum chunk passou no threshold do reranker (Score >= {RERANKER_SCORE_THRESHOLD}).")
                            justification = f"Nenhum trecho considerado suficientemente relevante após re-ranking (limite={RERANKER_SCORE_THRESHOLD})."
                            metrics["claims_with_no_reranked_context"] += 1; metrics["llm_validation_failures"] += 1
                        else:
                            logging.info(f"Selecionados {len(selected_chunks_for_llm_data)} chunks após reranking e threshold.")
                            if selected_chunks_for_llm_data: claim_top_rerank_score = selected_chunks_for_llm_data[0]['rerank_score']
                            used_context_for_llm = [chunk['text'] for chunk in selected_chunks_for_llm_data]
                            context_for_validator = {'documents': [used_context_for_llm]}
                            logging.info("Validando alegação com LLM..."); llm_start_time = time.time()
                            result = validate_claim_with_llm(claim_text, context_for_validator)
                            llm_time = time.time() - llm_start_time; metrics["total_llm_latency"] += llm_time
                            logging.debug(f"LLM validation took {llm_time:.2f}s")
                            if result is None:
                                logging.error("Falha na validação do LLM (retornou None). Marcando como Erro.")
                                justification = "Falha interna durante a validação do LLM (ver logs do validator)."; metrics["llm_validation_failures"] += 1
                            else:
                                status = result.get("Resultado", "Erro"); justification = result.get("Justificativa", "Erro ao parsear justificativa.")
                                claim_input_tokens = result.get("input_tokens", 0); claim_output_tokens = result.get("output_tokens", 0)
                                metrics["total_input_tokens"] += claim_input_tokens; metrics["total_output_tokens"] += claim_output_tokens
                                if status == "Erro" and ("Falha ao parsear" in justification or "Formato inesperado" in justification):
                                    metrics["llm_parsing_failures"] += 1; metrics["llm_validation_failures"] += 1
                                elif status == "Erro": metrics["llm_validation_failures"] += 1
            claim_processing_time = time.time() - claim_processing_start_time
            validation_results.append({
                "claim_index": claim_index, "claim_text": claim_text, "status": status, "justification": justification,
                "input_tokens": claim_input_tokens, "output_tokens": claim_output_tokens,
                "claim_processing_time": claim_processing_time, "top_rerank_score": claim_top_rerank_score,
                "used_context_for_llm": used_context_for_llm
            })
            metrics["processed_claims"] += 1
            logging.info(f"Resultado Alegação (Sentença) {claim_index}: {status}")
            logging.info(f"Tempo para Alegação (Sentença) {claim_index}: {claim_processing_time:.2f}s (Rec: {retrieval_time:.2f}s, Rerank: {rerank_time:.2f}s, LLM: {llm_time:.2f}s)")
        metrics["total_latency_seconds"] = time.time() - pipeline_start_time
        logging.info(f"Validação de todas as {metrics['processed_claims']}/{metrics['total_claims']} alegações (sentenças) concluída em {metrics['total_latency_seconds']:.2f} segundos.")
        return validation_results, metrics

    def save_results(results: list, acordao_path: str, output_path: str):
        logging.info(f"Salvando {len(results)} resultados em formato TXT: {output_path}")
        try:
            process_name = os.path.basename(acordao_path).split('.')[0]
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Processo {process_name}.\n")
                for res in results:
                    f.write(f"Reivindicação {res['claim_index']} (Sentença): \"{res['claim_text']}\" → {res['status']}.\n")
                    if res['status'] == 'Incorreta' and res['justification'] and res['justification'] != 'N/A':
                        f.write(f" Justificativa: {res['justification']}\n")
                    f.write("\n")
            logging.info("Resultados TXT salvos com sucesso.")
        except Exception as e: logging.error(f"Falha ao salvar arquivo de saída TXT: {e}", exc_info=True)

    def save_results_csv(results: List[Dict[str, Any]], output_filename: str):
        if not results: logging.warning("Nenhum resultado para salvar em CSV."); return
        logging.info(f"Salvando {len(results)} resultados detalhados em CSV: {output_filename}")
        fieldnames = ['ID', 'Sentenca', 'Prediction', 'Confianca', 'numero_tokens', 'runtime_seconds']
        try:
            with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                for item in results:
                    num_tokens = item.get('input_tokens', 0) + item.get('output_tokens', 0)
                    confidence_score = item.get('top_rerank_score')
                    writer.writerow({
                        'ID': item.get('claim_index', ''), 'Sentenca': item.get('claim_text', ''),
                        'Prediction': item.get('status', 'Erro'),
                        'Confianca': f"{confidence_score:.4f}" if confidence_score is not None else "N/A",
                        'numero_tokens': num_tokens, 'runtime_seconds': f"{item.get('claim_processing_time', 0):.2f}"
                    })
            logging.info(f"Resultados CSV salvos com sucesso em {output_filename}")
        except Exception as e: logging.error(f"Falha ao salvar arquivo CSV: {e}", exc_info=True)

    def save_justifications_csv(results: List[Dict[str, Any]], output_filename: str):
        if not results: logging.warning("Nenhum resultado para salvar justificativas em CSV."); return
        logging.info(f"Salvando {len(results)} justificativas e contextos em CSV: {output_filename}")
        fieldnames = ['ID', 'Sentenca', 'Prediction', 'Justificativa', 'ContextoUtilizado']
        try:
            output_dir = os.path.dirname(output_filename); 
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                for item in results:
                    context_str = " [SEP_CONTEXTO] ".join(item.get('used_context_for_llm', []))
                    writer.writerow({
                        'ID': item.get('claim_index', ''), 'Sentenca': item.get('claim_text', ''),
                        'Prediction': item.get('status', 'Erro'), 'Justificativa': item.get('justification', 'N/A'),
                        'ContextoUtilizado': context_str
                    })
            logging.info(f"Justificativas e contextos CSV salvos com sucesso em {output_filename}")
        except Exception as e: logging.error(f"Falha ao salvar arquivo CSV de justificativas e contextos: {e}", exc_info=True)

    def print_metrics(metrics: dict):
        logging.info("--- Métricas de Execução ---"); print("\n--- Métricas de Execução ---")
        total_claims = metrics.get("total_claims", 0); processed_claims = metrics.get("processed_claims", 0)
        if processed_claims == 0: print("Nenhuma alegação (sentença) foi processada."); logging.warning("Nenhuma alegação (sentença) foi processada para cálculo de métricas."); return
        total_latency = metrics.get("total_latency_seconds", 0); avg_latency_per_claim = total_latency / processed_claims if processed_claims else 0
        avg_retrieval_latency = metrics.get("total_retrieval_latency", 0) / processed_claims if processed_claims else 0
        avg_rerank_latency = metrics.get("total_rerank_latency", 0) / processed_claims if processed_claims else 0
        avg_llm_latency = metrics.get("total_llm_latency", 0) / processed_claims if processed_claims else 0
        print(f"Latência Total: {total_latency:.2f} segundos"); print(f"Latência Média por Alegação (Sentença): {avg_latency_per_claim:.2f} segundos")
        print(f"  - Média Recuperação (Retriever): {avg_retrieval_latency:.2f} segundos"); print(f"  - Média Re-ranking: {avg_rerank_latency:.2f} segundos"); print(f"  - Média Validação (LLM): {avg_llm_latency:.2f} segundos")
        logging.info(f"Latência: Total={total_latency:.2f}s, Média/Alegação={avg_latency_per_claim:.2f}s (Retriever={avg_retrieval_latency:.2f}s, Rerank={avg_rerank_latency:.2f}s, LLM={avg_llm_latency:.2f}s)")
        total_input_tokens = metrics.get("total_input_tokens", 0); total_output_tokens = metrics.get("total_output_tokens", 0); total_tokens = total_input_tokens + total_output_tokens
        avg_input_tokens = total_input_tokens / processed_claims if processed_claims else 0; avg_output_tokens = total_output_tokens / processed_claims if processed_claims else 0
        print(f"\nUso de Tokens LLM Total: {total_tokens}"); print(f"  - Entrada Total: {total_input_tokens}"); print(f"  - Saída Total: {total_output_tokens}")
        print(f"Uso de Tokens LLM Médio por Alegação (Sentença):"); print(f"  - Média Entrada: {avg_input_tokens:.1f}"); print(f"  - Média Saída: {avg_output_tokens:.1f}")
        logging.info(f"Tokens LLM: Total={total_tokens} (In={total_input_tokens}, Out={total_output_tokens}), Média/Alegação (In={avg_input_tokens:.1f}, Out={avg_output_tokens:.1f})")
        retrieval_failures = metrics.get("retrieval_failures", 0); llm_validation_failures = metrics.get("llm_validation_failures", 0)
        llm_parsing_failures = metrics.get("llm_parsing_failures", 0); claims_no_reranked = metrics.get("claims_with_no_reranked_context", 0)
        success_rate = ((processed_claims - llm_validation_failures) / processed_claims * 100) if processed_claims else 0
        print(f"\nConfiabilidade (em {processed_claims} alegações/sentenças processadas):"); print(f"  - Taxa de Sucesso (Validação Completa): {success_rate:.1f}%")
        print(f"  - Falhas de Recuperação Inicial: {retrieval_failures}"); print(f"  - Alegações Sem Contexto Pós-Rerank/Threshold: {claims_no_reranked}")
        print(f"  - Falhas de Validação (LLM Total): {llm_validation_failures}"); print(f"    - Falhas de Parsing Resposta LLM: {llm_parsing_failures}")
        logging.info(f"Confiabilidade: Sucesso={success_rate:.1f}%, Falhas Retriever={retrieval_failures}, Sem Ctx Pós-Rerank={claims_no_reranked}, Falhas LLM={llm_validation_failures} (Parsing={llm_parsing_failures})"); print("----------------------------")

    def print_results_summary(results: list):
        print("\n--- Resumo da Validação (Sentenças) ---")
        if not results: print("Nenhum resultado de validação para exibir."); logging.warning("Nenhum resultado de validação disponível para exibir no console."); return
        for res in results:
            claim_text_short = res.get('claim_text', '[VAZIA]')
            if len(claim_text_short) > 100: claim_text_short = claim_text_short[:100] + "..."
            status = res.get('status', 'Erro'); justification = res.get('justification', '')
            print(f"  - Alegação (Sentença) {res.get('claim_index', '?')}: \"{claim_text_short}\" -> {status}")
            if status == 'Incorreta' and justification and justification != 'N/A' and "Erro" not in justification: print(f"    Justificativa: {justification}")
        print("---------------------------\n")

    main() 