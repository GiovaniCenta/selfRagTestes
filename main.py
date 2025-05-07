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
        # Updated field names to reflect that this is the 'final' prediction
        fieldnames = ['ID_Claim_Resumo', 'Processo_ID', 'Sentenca_Claim', 
                      'Prediction_Final', 'Confianca_Final']
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for result in all_claim_results:
                writer.writerow({
                    'ID_Claim_Resumo': result['claim_id'],
                    'Processo_ID': processo_id,
                    'Sentenca_Claim': result['claim_text'],
                    'Prediction_Final': result['predicted_label'], # Changed from Prediction_Verificador
                    'Confianca_Final': f"{result['confidence']:.4f}" # Changed from Confianca_Verificador
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
        # Added 'Classifier_Votes_Details' for more detailed output if needed
        fieldnames = ['ID_Claim_Resumo', 'Processo_ID', 'Sentenca_Claim', 
                      'Prediction_Final', 'Confianca_Final', 'Justificativa_LLM_Ou_Classificador', 
                      'Contexto_Utilizado_LLM', 'Detalhes_Votos_Classificador'] 
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for result in all_claim_results:
                context_str = " [SEP_CONTEXTO] ".join(result.get('context_for_llm', []))
                # Serialize classifier_votes_details for CSV if it exists
                classifier_votes_str = str(result.get('classifier_votes', [])) # Simple string representation

                writer.writerow({
                    'ID_Claim_Resumo': result['claim_id'],
                    'Processo_ID': processo_id,
                    'Sentenca_Claim': result['claim_text'],
                    'Prediction_Final': result['predicted_label'], # This is now the final prediction
                    'Confianca_Final': f"{result['confidence']:.4f}",
                    'Justificativa_LLM_Ou_Classificador': result['rationale'], # This contains LLM rationale or classifier unanimity message
                    'Contexto_Utilizado_LLM': context_str,
                    'Detalhes_Votos_Classificador': classifier_votes_str
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
        final_predicted_label = "ERRO"
        final_confidence = 0.0
        final_rationale = "Processamento não concluído ou falhou."
        context_for_llm = [] # Context actually used by LLM (if called)
        classifier_votes_details = [] # To store (label, score) for each classifier vote
        skip_llm = False

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
                # Prepare context for LLM (top N) and for classifier voting
                # This context will be used by LLM if it's called.
                context_for_llm = [doc.get("text", "") for doc in top_n_reranked_docs if doc.get("text")]

                # --- 5a. Classifier Voting Stage ---
                logging.info(f"Step 5a (Claim {claim_id}): Performing classifier voting with {len(top_n_reranked_docs)} contexts...")
                classifier_votes = []
                classifier_confidences = []

                for reranked_doc in top_n_reranked_docs:
                    doc_text = reranked_doc.get("text")
                    if not doc_text:
                        logging.warning(f"Claim {claim_id}: A reranked document has no 'text' field. Skipping for classifier voting.")
                        continue
                    
                    clf_label, clf_prob = classify_claim(current_query_claim, doc_text)
                    classifier_votes.append(clf_label)
                    classifier_confidences.append(clf_prob)
                    classifier_votes_details.append({"doc_text_preview": doc_text[:100], "label": clf_label, "score": clf_prob})
                
                logging.info(f"Claim {claim_id}: Classifier votes: {classifier_votes}, Confidences: {[round(c, 4) for c in classifier_confidences]}")

                # --- 5b. Analyze Votes & Conditional LLM Skip ---
                if not classifier_votes: # Should not happen if top_n_reranked_docs was not empty and had text
                    logging.warning(f"Claim {claim_id}: No classifier votes recorded despite having reranked docs. Defaulting to LLM.")
                    skip_llm = False 
                else:
                    num_votes = len(classifier_votes)
                    # Check for unanimity (Requires at least 1 vote)
                    # We could add a threshold, e.g., requires at least min_votes_for_unanimity
                    is_unanimous = len(set(classifier_votes)) == 1 

                    if is_unanimous:
                        unanimous_label = classifier_votes[0]
                        avg_confidence = sum(classifier_confidences) / num_votes
                        
                        final_predicted_label = unanimous_label
                        final_confidence = avg_confidence
                        final_rationale = f"Classificador unânime ({num_votes} votos): {unanimous_label}. LLM não consultado."
                        logging.info(f"Claim {claim_id}: Unanimous classifier decision: {final_predicted_label} with avg confidence {final_confidence:.4f}. Skipping LLM.")
                        skip_llm = True
                    else:
                        # Mixed votes or other non-unanimous conditions
                        logging.info(f"Claim {claim_id}: Classifier votes are mixed or condition for skipping LLM not met. Proceeding to LLM explanation.")
                        skip_llm = False
                        # LLM will determine final_predicted_label, final_rationale.
                        # final_confidence will be from LLM (e.g., 1.0)

                # --- 6. Generate explanation using LLM (if not skipped) --- 
                if not skip_llm:
                    if not context_for_llm: # Should not happen if top_n_reranked_docs was populated
                        logging.warning(f"Claim {claim_id}: No text found in reranked documents for generating LLM explanation, though LLM was supposed to run.")
                        final_predicted_label = "ERRO" # Or keep classifier's if available/sensible
                        final_rationale = "Erro: Contexto para LLM estava vazio, embora o LLM devesse ser executado."
                        final_confidence = 0.0
                    else:
                        logging.info(f"Step 6 (Claim {claim_id}): Generating LLM prediction and explanation using top {len(context_for_llm)} reranked documents...")
                        try:
                            # generate_explanation now returns (prediction, rationale)
                            llm_prediction, llm_rationale = generate_explanation(
                                current_query_claim, 
                                context_for_llm, 
                                max_new_tokens=args.explainer_max_tokens, 
                                temperature=args.explainer_temperature
                            )
                            final_predicted_label = llm_prediction
                            final_rationale = llm_rationale
                            # If LLM provides prediction, we can set a nominal confidence or parse if available
                            final_confidence = 1.0 if llm_prediction not in ["ERRO_DE_PARSING", "ERRO"] else 0.0 
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
            "context_for_llm": context_for_llm, # Full context list intended for LLM
            "classifier_votes": classifier_votes_details # Optional: For detailed logging/debugging
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

if __name__ == "__main__":
    run_rag_pipeline()
