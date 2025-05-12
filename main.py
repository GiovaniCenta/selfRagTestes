import logging
import argparse
import os
import csv # Added for CSV output
import datetime # Added for timestamp in filenames
import json # Para salvar resultados do Self-RAG
from dotenv import load_dotenv

# Import new modules
from src.data_loader import pdf_to_docs, load_document, split_text_into_paragraphs, _split_paragraphs_into_sentences # Added functions
from src.indexer import upsert_items
from src.retriever import search as retrieve_candidates
from src.reranker import rank as rerank_candidates
# from src.verifier import classify as classify_claim # Replaced validator.py - REMOVING THIS
from src.llm_explainer import explain as generate_explanation # New module
from src.self_rag import answer_with_self_rag # Novo módulo para Self-RAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - MAIN - %(levelname)s - %(message)s')

# --- Helper function to create output directory ---
def _ensure_output_dir(output_dir="results"):
    """Ensures the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# --- Functions to save results --- 
def save_summary_txt(all_claim_results, output_filepath, pdf_file_name, resumo_file_name, processo_id):
    """Saves a human-readable summary of claim validation results to a TXT file."""
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Resumo da Validação de Alegações\n")
            f.write(f"Data da Execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Documento Principal (Acórdão): {pdf_file_name}\n")
            f.write(f"Arquivo de Resumo (Alegações): {resumo_file_name}\n")
            f.write(f"Processo ID: {processo_id}\n")
            f.write("="*50 + "\n\n")

            if not all_claim_results:
                f.write("Nenhuma alegação foi processada ou nenhuma resultado disponível.\n")
                return

            for result in all_claim_results:
                f.write(f"Alegação ID: {result['claim_id']}\n")
                f.write(f"  Texto: {result['claim_text']}\n")
                f.write(f"  Predição do LLM: {result['predicted_label']}\n") # Changed from Verificador
                f.write(f"  Confiança: {result['confidence']:.4f}\n")
                # Correctly handle backslashes in f-string expression
                justificativa_formatada = result['rationale'].replace('\n', '\n    ')
                f.write(f"  Justificativa do LLM:\n    {justificativa_formatada}\n") # Removed trailing space and newline in f-string for cleaner output
                f.write("-"*30 + "\n")
        logging.info(f"Resumo TXT salvo em: {output_filepath}")
    except Exception as e:
        logging.error(f"Falha ao salvar resumo TXT em {output_filepath}: {e}", exc_info=True)

def save_results_csv(all_claim_results, output_filepath, processo_id):
    """Saves detailed claim validation results to a CSV file."""
    if not all_claim_results:
        logging.warning("Nenhum resultado para salvar no CSV de resultados.")
        return
    try:
        # Updated field names to reflect that this is the LLM's prediction
        fieldnames = ['ID_Claim_Resumo', 'Processo_ID', 'Sentenca_Claim', 
                      'Prediction_LLM', 'Confianca_LLM'] # Renamed from _Final
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for result in all_claim_results:
                writer.writerow({
                    'ID_Claim_Resumo': result['claim_id'],
                    'Processo_ID': processo_id,
                    'Sentenca_Claim': result['claim_text'],
                    'Prediction_LLM': result['predicted_label'], 
                    'Confianca_LLM': f"{result['confidence']:.4f}"
                })
        logging.info(f"Resultados CSV salvos em: {output_filepath}")
    except Exception as e:
        logging.error(f"Falha ao salvar resultados CSV em {output_filepath}: {e}", exc_info=True)

def save_justifications_csv(all_claim_results, output_filepath, processo_id):
    """Saves justifications and context to a CSV file."""
    if not all_claim_results:
        logging.warning("Nenhum resultado para salvar no CSV de justificativas.")
        return
    try:
        # Simplified field names, removing classifier details
        fieldnames = ['ID_Claim_Resumo', 'Processo_ID', 'Sentenca_Claim', 
                      'Prediction_LLM', 'Confianca_LLM', 'Justificativa_LLM', 
                      'Contexto_Utilizado_LLM'] 
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for result in all_claim_results:
                context_str = " [SEP_CONTEXTO] ".join(result.get('context_for_llm', []))
                writer.writerow({
                    'ID_Claim_Resumo': result['claim_id'],
                    'Processo_ID': processo_id,
                    'Sentenca_Claim': result['claim_text'],
                    'Prediction_LLM': result['predicted_label'], 
                    'Confianca_LLM': f"{result['confidence']:.4f}",
                    'Justificativa_LLM': result['rationale'], 
                    'Contexto_Utilizado_LLM': context_str
                })
        logging.info(f"Justificativas CSV salvas em: {output_filepath}")
    except Exception as e:
        logging.error(f"Falha ao salvar justificativas CSV em {output_filepath}: {e}", exc_info=True)

def run_rag_pipeline():
    """Main function to run the STF-RAG pipeline."""
    # --- Load Environment Variables --- 
    load_dotenv() # Load from .env file in the project root
    logging.info("Loaded environment variables from .env if present.")

    # --- Argument Parsing --- 
    parser = argparse.ArgumentParser(description="STF-RAG Pipeline: Index, Retrieve, Rerank, Classify, and Explain claims from a Resumo file.")
    parser.add_argument(
        "--pdf_file", 
        type=str, 
        default="data/Acórdão 733 de 2025 Plenário.pdf", 
        help="Path to the PDF file (Acórdão) to process and index. Defaults to data/Acordao_Default.pdf"
    )
    parser.add_argument(
        "--resumo_file", 
        type=str, 
        default="data/Acórdão 764-2025 resumos.txt", 
        help="Path to the Resumo TXT file containing claims to validate. Defaults to data/Resumo_Default.txt"
    )
    # New arguments for output files
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output files.")
    parser.add_argument("--txt_output_file", type=str, default="", help="Filename for the TXT summary output. If empty, defaults to [ProcessoID]_summary.txt.")
    parser.add_argument("--csv_results_file", type=str, default="", help="Filename for the CSV results output. If empty, defaults to [ProcessoID]_resultados.csv.")
    parser.add_argument("--csv_justifications_file", type=str, default="", help="Filename for the CSV justifications output. If empty, defaults to [ProcessoID]_justificativas.csv.")


    # Retrieval and Reranking parameters
    parser.add_argument("--retrieval_k", type=int, default=20, help="Number of candidates to retrieve initially for each claim.")
    parser.add_argument("--reranker_n", type=int, default=5, help="Number of top candidates to keep after reranking for each claim.")

    # LLM Explainer parameters
    parser.add_argument("--explainer_max_tokens", type=int, default=150, help="Max new tokens for LLM explanation for each claim.")
    parser.add_argument("--explainer_temperature", type=float, default=0.1, help="Temperature for LLM explanation generation for each claim.")

    args = parser.parse_args()

    logging.info(f"Starting RAG pipeline for PDF: {args.pdf_file} with Resumo: {args.resumo_file}")
    output_directory = _ensure_output_dir(args.output_dir)

    # --- Check if default files exist if they are being used --- 
    if not os.path.exists(args.pdf_file):
        logging.error(f"Error: PDF file not found at '{args.pdf_file}'. Please provide a valid path or ensure the default file exists.")
        return
    if not os.path.exists(args.resumo_file):
        logging.error(f"Error: Resumo file not found at '{args.resumo_file}'. Please provide a valid path or ensure the default file exists.")
        return

    # --- 1. Load Acórdão data from PDF and generate embeddings --- 
    logging.info("Step 1: Processing Acórdão PDF and generating embeddings...")
    docs_with_embeddings = pdf_to_docs(args.pdf_file)
    if not docs_with_embeddings:
        logging.error(f"Failed to process PDF {args.pdf_file} or no documents extracted. Exiting.")
        return
    
    processo_id_from_pdf = docs_with_embeddings[0].get("processo")
    if not processo_id_from_pdf:
        logging.error("Could not determine processo_id from the processed PDF documents. Exiting.")
        return
    logging.info(f"Determined Processo ID from PDF: {processo_id_from_pdf}")

    # Define output filenames based on processo_id or user args
    pdf_basename = os.path.basename(args.pdf_file)
    resumo_basename = os.path.basename(args.resumo_file)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Use os.path.basename on user-provided filenames to prevent duplication
    txt_output_filename = os.path.basename(args.txt_output_file) if args.txt_output_file else f"{processo_id_from_pdf}_summary_{timestamp}.txt"
    csv_results_filename = os.path.basename(args.csv_results_file) if args.csv_results_file else f"{processo_id_from_pdf}_resultados_{timestamp}.csv"
    csv_justifications_filename = os.path.basename(args.csv_justifications_file) if args.csv_justifications_file else f"{processo_id_from_pdf}_justificativas_{timestamp}.csv"

    # Join the potentially stripped filename with the guaranteed output directory
    full_txt_path = os.path.join(output_directory, txt_output_filename)
    full_csv_results_path = os.path.join(output_directory, csv_results_filename)
    full_csv_justifications_path = os.path.join(output_directory, csv_justifications_filename)

    # --- 2. Index Acórdão documents into Cosmos DB --- 
    logging.info("Step 2: Indexing Acórdão documents into Cosmos DB...")
    try:
        upsert_items(docs_with_embeddings)
        logging.info("Acórdão document indexing complete.")
    except Exception as e:
        logging.error(f"Failed during Acórdão document indexing: {e}", exc_info=True)
        return 

    # --- Load and Process Resumo File --- 
    logging.info(f"Loading and processing Resumo file: {args.resumo_file}")
    resumo_content = load_document(args.resumo_file)
    if not resumo_content or not isinstance(resumo_content, str):
        logging.error(f"Failed to load or read Resumo file as text: {args.resumo_file}. Exiting.")
        return
    
    resumo_paragraphs = split_text_into_paragraphs(resumo_content)
    if not resumo_paragraphs:
        logging.warning(f"No paragraphs found in Resumo file: {args.resumo_file}. No claims to process.")
        return
        
    claims_from_resumo = _split_paragraphs_into_sentences(resumo_paragraphs)
    if not claims_from_resumo:
        logging.warning(f"No sentences (claims) extracted from Resumo file: {args.resumo_file}. No claims to process.")
        return
    
    logging.info(f"Extracted {len(claims_from_resumo)} claims from Resumo file.")

    # --- Loop through each claim and run RAG steps 3-7 --- 
    all_claim_processing_results = [] # To store results for each claim

    for i, current_query_claim in enumerate(claims_from_resumo):
        claim_id = i + 1
        logging.info(f"\nProcessing Claim {claim_id}/{len(claims_from_resumo)}: '{current_query_claim[:100]}...'")
        
        # Initialize final results for the claim
        final_predicted_label = "ERRO" # Will be set by LLM
        final_confidence = 0.0    # Will be set based on LLM output
        final_rationale = "Processamento não concluído ou falhou."
        context_for_llm = [] # Context actually used by LLM

        # --- 3. Retrieve candidate documents for the current claim --- 
        logging.info(f"Step 3 (Claim {claim_id}): Retrieving top {args.retrieval_k} candidates for Processo ID: {processo_id_from_pdf}...")
        candidate_docs = retrieve_candidates(current_query_claim, processo_id_from_pdf, k=args.retrieval_k)
        
        if not candidate_docs:
            logging.warning(f"Claim {claim_id}: No candidate documents retrieved. Marking as NÃO CONFERE.")
            final_predicted_label = "NÃO CONFERE"
            final_confidence = 0.0 # Or 1.0 for certainty of "NÃO CONFERE" due to no docs
            final_rationale = "Nenhum documento relevante encontrado no acórdão para a afirmação fornecida."
            # skip_llm is already False, LLM won't be called if no docs.
        else:
            # --- 4. Rerank candidate documents for the current claim --- 
            logging.info(f"Step 4 (Claim {claim_id}): Reranking top {len(candidate_docs)} candidates to select {args.reranker_n}...")
            top_n_reranked_docs = rerank_candidates(current_query_claim, candidate_docs, n=args.reranker_n)
            
            if not top_n_reranked_docs:
                logging.warning(f"Claim {claim_id}: No documents survived reranking. Marking as NÃO CONFERE.")
                final_predicted_label = "NÃO CONFERE"
                final_confidence = 0.0 # Or 1.0
                final_rationale = "Os documentos encontrados inicialmente não foram considerados suficientemente relevantes após o re-rankeamento."
                # skip_llm is already False, LLM won't be called.
            else:
                # Prepare context for LLM (top N)
                # This context will be used by LLM if it's called.
                context_for_llm = [doc.get("text", "") for doc in top_n_reranked_docs if doc.get("text")]

                # --- Step 5 (was 6). Generate prediction and explanation using LLM (always called if context exists) ---
                if not context_for_llm: 
                    logging.warning(f"Claim {claim_id}: No text found in reranked documents for generating LLM explanation.")
                    final_predicted_label = "ERRO_CONTEXTO_LLM"
                    final_rationale = "Erro: Contexto para LLM estava vazio após rerankamento."
                    final_confidence = 0.0
                else:
                    logging.info(f"Step 5 (Claim {claim_id}): Generating LLM prediction and explanation using top {len(context_for_llm)} reranked documents...")
                    try:
                        llm_prediction, llm_rationale = generate_explanation(
                            current_query_claim, 
                            context_for_llm, 
                            max_new_tokens=args.explainer_max_tokens, 
                            temperature=args.explainer_temperature
                        )
                        final_predicted_label = llm_prediction
                        final_rationale = llm_rationale
                        final_confidence = 1.0 if llm_prediction not in ["ERRO_DE_PARSING", "ERRO", "ERRO_CONTEXTO_LLM"] else 0.0 
                        logging.info(f"Claim {claim_id}: LLM determined label: {final_predicted_label}")
                    except Exception as e_llm:
                        logging.error(f"Claim {claim_id}: LLM explanation step failed: {e_llm}", exc_info=True)
                        final_predicted_label = "ERRO_LLM"
                        final_rationale = f"Falha na geração da explicação pelo LLM: {str(e_llm)}"
                        final_confidence = 0.0
                # If skip_llm was True, final_predicted_label, final_confidence, final_rationale are already set from classifier.

        # --- Store results for this claim --- 
        all_claim_processing_results.append({
            "claim_id": claim_id,
            "claim_text": current_query_claim,
            "predicted_label": final_predicted_label,
            "confidence": final_confidence,
            "rationale": final_rationale,
            "context_for_llm": context_for_llm # Full context list intended for LLM
        })

        # --- Print Live Output for the current claim --- 
        print("\n" + "="*20 + f" RAG Pipeline Result for Claim {claim_id} " + "="*20)
        print(f"Input PDF: {pdf_basename}")
        print(f"Processo ID: {processo_id_from_pdf}")
        print(f"Resumo File: {resumo_basename}")
        print(f"Claim {claim_id} Text: {current_query_claim}")
        print("-"*50)
        print(f"Predicted Label: {final_predicted_label}")
        print(f"Confidence: {final_confidence:.4f}")
        print(f"Rationale (Explanation):\n{final_rationale}")
        print("="*65 + "\n")
    
    # --- Save all results to files after processing all claims --- 
    if all_claim_processing_results:
        logging.info(f"Saving all {len(all_claim_processing_results)} processed claim results to files...")
        save_summary_txt(all_claim_processing_results, full_txt_path, pdf_basename, resumo_basename, processo_id_from_pdf)
        save_results_csv(all_claim_processing_results, full_csv_results_path, processo_id_from_pdf)
        save_justifications_csv(all_claim_processing_results, full_csv_justifications_path, processo_id_from_pdf)
    else:
        logging.warning("No claims were processed successfully, skipping file saving.")

    logging.info(f"Finished processing all {len(claims_from_resumo)} claims from {args.resumo_file}.")

def run_self_rag_qa():
    """Function to run the Self-RAG Q&A system."""
    # --- Load Environment Variables --- 
    load_dotenv() # Load from .env file in the project root
    logging.info("Loaded environment variables from .env if present.")
    
    # --- Argument Parsing --- 
    parser = argparse.ArgumentParser(description="Self-RAG Q&A: Utilizando RAG com auto-refinamento para perguntas e respostas.")
    parser.add_argument(
        "--pdf_file", 
        type=str, 
        default="data/Acórdão 733 de 2025 Plenário.pdf", 
        help="Caminho para o PDF principal a ser processado e indexado."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="", 
        help="Pergunta a ser respondida. Se vazia, será solicitada interativamente."
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Modo interativo para fazer múltiplas perguntas."
    )
    
    # Parâmetros do Self-RAG
    parser.add_argument("--output_dir", type=str, default="results", help="Diretório para salvar resultados.")
    parser.add_argument("--max_attempts", type=int, default=3, help="Número máximo de tentativas de refinamento para cada pergunta.")
    parser.add_argument("--initial_k", type=int, default=10, help="Número de documentos iniciais a recuperar.")
    parser.add_argument("--reranker_n", type=int, default=5, help="Número de documentos após reranking.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperatura do LLM para geração de respostas.")
    parser.add_argument("--max_tokens", type=int, default=200, help="Número máximo de tokens por resposta gerada.")
    parser.add_argument("--save_results", action="store_true", help="Salvar resultados em arquivo JSON.")
    
    args = parser.parse_args()
    
    # Verificar e criar diretório de saída
    output_directory = _ensure_output_dir(args.output_dir)
    
    # --- Verificar arquivo PDF ---
    if not os.path.exists(args.pdf_file):
        logging.error(f"Arquivo PDF não encontrado: '{args.pdf_file}'. Forneça um caminho válido.")
        return
    
    # --- 1. Carregar e indexar PDF ---
    logging.info(f"Processando PDF: {args.pdf_file}")
    docs_with_embeddings = pdf_to_docs(args.pdf_file)
    if not docs_with_embeddings:
        logging.error(f"Falha ao processar o PDF {args.pdf_file} ou nenhum documento extraído. Saindo.")
        return
    
    processo_id_from_pdf = docs_with_embeddings[0].get("processo")
    if not processo_id_from_pdf:
        logging.error("Não foi possível determinar o ID do processo a partir dos documentos processados. Saindo.")
        return
    logging.info(f"ID do Processo identificado: {processo_id_from_pdf}")
    
    # --- 2. Indexar documentos no Cosmos DB ---
    logging.info("Indexando documentos...")
    try:
        upsert_items(docs_with_embeddings)
        logging.info("Indexação dos documentos concluída.")
    except Exception as e:
        logging.error(f"Falha durante a indexação dos documentos: {e}", exc_info=True)
        return
    
    # --- 3. Modo de execução: Interativo ou Consulta Única ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    pdf_basename = os.path.basename(args.pdf_file)
    
    if args.interactive:
        # --- Modo interativo ---
        all_qa_results = []
        query_counter = 1
        
        print("\n" + "="*30 + " MODO INTERATIVO " + "="*30)
        print(f"PDF indexado: {pdf_basename}")
        print(f"ID do Processo: {processo_id_from_pdf}")
        print("Digite suas perguntas. Para sair, digite 'sair' ou 'exit'.\n")
        
        while True:
            # Solicitar pergunta ao usuário
            user_query = input("\nSua pergunta: ")
            if user_query.lower() in ['sair', 'exit', 'quit']:
                break
                
            if not user_query.strip():
                print("Pergunta vazia. Por favor, tente novamente.")
                continue
            
            print("\nProcessando sua pergunta...")
            
            # Processar a pergunta
            qa_result = answer_with_self_rag(
                user_query,
                processo_id=processo_id_from_pdf,
                initial_k=args.initial_k,
                reranker_n=args.reranker_n,
                max_attempts=args.max_attempts,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            # Adicionar informações adicionais ao resultado
            qa_result["query"] = user_query
            qa_result["query_id"] = query_counter
            all_qa_results.append(qa_result)
            query_counter += 1
            
            # Exibir resposta
            print("\n" + "-"*60)
            print(f"Resposta: {qa_result['answer']}")
            print("-"*60)
            print(f"Qualidade: {qa_result['final_quality_score']}/10")
            print(f"Tentativas: {qa_result['total_attempts']}/{args.max_attempts}")
            print(f"Tokens totais: {qa_result['stats']['total_input_tokens'] + qa_result['stats']['total_output_tokens']}")
            print(f"Tokens adicionais (avaliação/refinamento): {qa_result['stats']['additional_tokens']}")
            print("-"*60)
        
        # Salvar resultados se solicitado
        if args.save_results and all_qa_results:
            results_filename = f"{processo_id_from_pdf}_qa_results_{timestamp}.json"
            results_filepath = os.path.join(output_directory, results_filename)
            
            try:
                with open(results_filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_qa_results, f, ensure_ascii=False, indent=2)
                print(f"\nResultados salvos em: {results_filepath}")
            except Exception as e:
                logging.error(f"Erro ao salvar resultados: {e}")
                print(f"Erro ao salvar resultados: {e}")
    
    else:
        # --- Modo consulta única ---
        query = args.query
        
        # Se a consulta não foi fornecida como argumento, solicitar
        if not query:
            query = input("Digite sua pergunta: ")
            if not query.strip():
                logging.error("Pergunta vazia. Saindo.")
                return
        
        logging.info(f"Processando pergunta: '{query}'")
        
        # Processar a pergunta
        qa_result = answer_with_self_rag(
            query,
            processo_id=processo_id_from_pdf,
            initial_k=args.initial_k,
            reranker_n=args.reranker_n,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Adicionar informações adicionais ao resultado
        qa_result["query"] = query
        qa_result["query_id"] = 1
        
        # Exibir resposta
        print("\n" + "="*30 + " RESULTADO " + "="*30)
        print(f"Pergunta: {query}")
        print("\n" + "-"*60)
        print(f"Resposta: {qa_result['answer']}")
        print("-"*60)
        print(f"Qualidade: {qa_result['final_quality_score']}/10")
        print(f"Tentativas: {qa_result['total_attempts']}/{args.max_attempts}")
        print(f"Tokens totais: {qa_result['stats']['total_input_tokens'] + qa_result['stats']['total_output_tokens']}")
        print(f"Tokens adicionais (avaliação/refinamento): {qa_result['stats']['additional_tokens']}")
        print("-"*60)
        
        # Salvar resultados se solicitado
        if args.save_results:
            results_filename = f"{processo_id_from_pdf}_qa_single_{timestamp}.json"
            results_filepath = os.path.join(output_directory, results_filename)
            
            try:
                with open(results_filepath, 'w', encoding='utf-8') as f:
                    json.dump([qa_result], f, ensure_ascii=False, indent=2)
                print(f"\nResultado salvo em: {results_filepath}")
            except Exception as e:
                logging.error(f"Erro ao salvar resultado: {e}")
                print(f"Erro ao salvar resultado: {e}")
    
    logging.info("Execução do Self-RAG Q&A concluída.")

if __name__ == "__main__":
    # Determinar qual pipeline executar
    parser = argparse.ArgumentParser(description="Sistema de RAG para validação de alegações ou perguntas e respostas.")
    parser.add_argument("--mode", type=str, choices=["validate", "qa"], default="validate", 
                        help="Modo de execução: 'validate' para validação de alegações, 'qa' para perguntas e respostas")
    
    args, remaining_args = parser.parse_known_args()
    
    # Ajustar sys.argv para passar os argumentos restantes para a função escolhida
    import sys
    sys.argv = [sys.argv[0]] + remaining_args
    
    if args.mode == "validate":
        run_rag_pipeline()
    elif args.mode == "qa":
        run_self_rag_qa()
