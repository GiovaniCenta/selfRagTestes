import logging
import argparse
import os
import csv # Added for CSV output
import datetime # Added for timestamp in filenames
from dotenv import load_dotenv

# Import new modules
from src.data_loader import pdf_to_docs, load_document, split_text_into_paragraphs, _split_paragraphs_into_sentences # Added functions
from src.indexer import upsert_items
from src.retriever import search as retrieve_candidates
from src.reranker import rank as rerank_candidates
from src.verifier import classify as classify_claim # Replaces validator.py
from src.llm_explainer import explain as generate_explanation # New module

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
                f.write(f"  Predição do Verificador: {result['predicted_label']}\n")
                f.write(f"  Confiança: {result['confidence']:.4f}\n")
                f.write(f"  Justificativa do LLM:\n    {result['rationale'].replace('\n', '\n    ')}\n")
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
        fieldnames = ['ID_Claim_Resumo', 'Processo_ID', 'Sentenca_Claim', 
                      'Prediction_Verificador', 'Confianca_Verificador']
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for result in all_claim_results:
                writer.writerow({
                    'ID_Claim_Resumo': result['claim_id'],
                    'Processo_ID': processo_id,
                    'Sentenca_Claim': result['claim_text'],
                    'Prediction_Verificador': result['predicted_label'],
                    'Confianca_Verificador': f"{result['confidence']:.4f}" # Format as string for consistency
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
        fieldnames = ['ID_Claim_Resumo', 'Processo_ID', 'Sentenca_Claim', 
                      'Prediction_Verificador', 'Justificativa_LLM', 'Contexto_Utilizado_LLM']
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for result in all_claim_results:
                # Context used for explainer was stored as a list of strings
                context_str = " [SEP_CONTEXTO] ".join(result.get('context_for_llm', []))
                writer.writerow({
                    'ID_Claim_Resumo': result['claim_id'],
                    'Processo_ID': processo_id,
                    'Sentenca_Claim': result['claim_text'],
                    'Prediction_Verificador': result['predicted_label'],
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

    # Define output filenames based on processo_id if not provided
    pdf_basename = os.path.basename(args.pdf_file)
    resumo_basename = os.path.basename(args.resumo_file)

    txt_output_filename = args.txt_output_file if args.txt_output_file else f"{processo_id_from_pdf}_summary_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    csv_results_filename = args.csv_results_file if args.csv_results_file else f"{processo_id_from_pdf}_resultados_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    csv_justifications_filename = args.csv_justifications_file if args.csv_justifications_file else f"{processo_id_from_pdf}_justificativas_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

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
        
        # Initialize default values for this claim's results
        predicted_label = "ERRO"
        probability = 0.0
        rationale = "Processamento não concluído ou falhou antes da geração da justificativa."
        context_for_llm = []

        # --- 3. Retrieve candidate documents for the current claim --- 
        logging.info(f"Step 3 (Claim {claim_id}): Retrieving top {args.retrieval_k} candidates for Processo ID: {processo_id_from_pdf}...")
        candidate_docs = retrieve_candidates(current_query_claim, processo_id_from_pdf, k=args.retrieval_k)
        
        if not candidate_docs:
            logging.warning(f"Claim {claim_id}: No candidate documents retrieved. Marking as NÃO CONFERE.")
            predicted_label = "NÃO CONFERE"
            probability = 0.0
            rationale = "Nenhum documento relevante encontrado no acórdão para a afirmação fornecida."
        else:
            # --- 4. Rerank candidate documents for the current claim --- 
            logging.info(f"Step 4 (Claim {claim_id}): Reranking top {len(candidate_docs)} candidates to select {args.reranker_n}...")
            top_n_reranked_docs = rerank_candidates(current_query_claim, candidate_docs, n=args.reranker_n)
            
            if not top_n_reranked_docs:
                logging.warning(f"Claim {claim_id}: No documents survived reranking. Marking as NÃO CONFERE.")
                predicted_label = "NÃO CONFERE"
                probability = 0.0
                rationale = "Os documentos encontrados inicialmente não foram considerados suficientemente relevantes após o re-rankeamento."
            else:
                context_for_llm = [doc.get("text", "") for doc in top_n_reranked_docs if doc.get("text")]
                # --- 5. Classify current claim based on top-1 reranked document --- 
                top_1_doc_text = top_n_reranked_docs[0].get("text")
                if not top_1_doc_text:
                    logging.error(f"Claim {claim_id}: Top reranked document has no 'text' field. Marking as ERRO.")
                    # predicted_label remains "ERRO"
                    rationale = "Erro interno ao processar o documento principal para classificação."
                else:
                    logging.info(f"Step 5 (Claim {claim_id}): Classifying claim using top-1 reranked document...")
                    predicted_label, probability = classify_claim(current_query_claim, top_1_doc_text)

                # --- 6. Generate explanation for the current claim --- 
                logging.info(f"Step 6 (Claim {claim_id}): Generating explanation using top {len(top_n_reranked_docs)} reranked documents...")
                if not context_for_llm:
                    logging.warning(f"Claim {claim_id}: No text found in reranked documents for generating explanation.")
                    rationale = "Não foi possível gerar uma explicação pois os documentos relevantes não continham texto para o LLM."
                else:
                    rationale = generate_explanation(
                        current_query_claim, 
                        context_for_llm, 
                        max_new_tokens=args.explainer_max_tokens, 
                        temperature=args.explainer_temperature
                    )

        # --- Store results for this claim --- 
        all_claim_processing_results.append({
            "claim_id": claim_id,
            "claim_text": current_query_claim,
            "predicted_label": predicted_label,
            "confidence": probability,
            "rationale": rationale,
            "context_for_llm": context_for_llm # Store the actual list of context strings used
        })

        # --- Print Live Output for the current claim --- 
        print("\n" + "="*20 + f" RAG Pipeline Result for Claim {claim_id} " + "="*20)
        print(f"Input PDF: {pdf_basename}")
        print(f"Processo ID: {processo_id_from_pdf}")
        print(f"Resumo File: {resumo_basename}")
        print(f"Claim {claim_id} Text: {current_query_claim}")
        print("-"*50)
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence: {probability:.4f}")
        print(f"Rationale (Explanation):\n{rationale}")
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

if __name__ == "__main__":
    run_rag_pipeline()
