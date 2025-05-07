import os
from PyPDF2 import PdfReader
import logging
import re # Import regex module
from typing import List, Tuple, Union, Dict, Optional, Any
import nltk # Added NLTK
from sentence_transformers import SentenceTransformer
import PyPDF2
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - LOADER - %(levelname)s - %(message)s')

# --- Constants --- 
# Updated Model Name
MODEL_NAME = "rufimelo/Legal-BERTimbau-sts-large"

# --- Cached Model Instance --- 
_embedder_instance: Optional[SentenceTransformer] = None

def get_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Loads or returns the cached SentenceTransformer model."""
    global _embedder_instance
    if _embedder_instance is None:
        logging.info(f"Loading embedding model: {model_name}")
        # Consider adding trust_remote_code=True if needed for this specific model
        _embedder_instance = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Embedding model loaded on device: {_embedder_instance.device}")
    return _embedder_instance

def _extract_processo_from_filename(filename: str) -> Optional[str]:
    """Extracts 'NNN-YYYY' pattern from filename if possible."""
    match = re.search(r'(\d+)[-_](\d{4})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    logging.warning(f"Could not extract 'processo' (NNN-YYYY) from filename: {filename}")
    return None

def _ensure_nltk_punkt():
    """Ensures that the NLTK 'punkt' resource is available."""
    try:
        nltk.data.find('tokenizers/punkt')
        logging.debug("NLTK 'punkt' resource found.")
    except nltk.downloader.DownloadError:
        logging.info("NLTK 'punkt' resource not found. Downloading...")
        nltk.download('punkt')
        logging.info("NLTK 'punkt' resource downloaded successfully.")
    except Exception as e:
        logging.error(f"Error checking/downloading NLTK 'punkt': {e}")
        # Depending on policy, you might want to re-raise or exit

_ensure_nltk_punkt() # Call this once when the module is loaded

def load_document(file_path: str) -> Union[str, List[Tuple[int, str]], None]:
    """
    Reads content of a text file (.txt) or a PDF file (.pdf).

    For TXT files, returns the entire content as a single string.
    For PDF files, returns list of tuples, where each tuple contains
    the 1-based page number and extracted text for that page.

    Args:
        file_path: The path to the file.

    Returns:
        - str: Full text content if the file is a .txt.
        - List[Tuple[int, str]]: List of (page_number, page_text) if the file is a .pdf.
        - None: If an error occurs or format not supported.
    """
    if not os.path.exists(file_path):
        logging.error(f"Error: File not found at '{file_path}'")
        return None

    _, file_extension = os.path.splitext(file_path)

    try:
        if file_extension.lower() == '.txt':
            # Read text file with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension.lower() == '.pdf':
            # Extract text from PDF page by page, keeping page numbers
            pages_content: List[Tuple[int, str]] = []
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                logging.info(f"Reading PDF: {file_path} with {len(reader.pages)} pages.")
                for i, page in enumerate(reader.pages):
                    page_num = i + 1 # Pages are 1-indexed
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            pages_content.append((page_num, page_text))
                        else:
                            # Still add entry for blank pages? Or skip? Skipping for now.
                            logging.warning(f"No text extracted from page {page_num} of {file_path}")
                    except Exception as e:
                        # Log errors during page extraction but continue if possible
                        logging.warning(f"Error extracting text from page {page_num} of {file_path}: {e}")
                        # Optionally add empty entry on error:
                        # pages_content.append((page_num, ""))

                if not pages_content:
                    logging.warning(f"Could not extract any text from PDF: {file_path}")
                    # Return empty list instead of None if file exists but no text extracted
                    return []

            return pages_content
        else:
            # Handle unsupported file formats
            logging.warning(f"Unsupported file format: '{file_extension}'. Only .txt and .pdf are accepted.")
            return None
    except Exception as e:
        # Catch-all for other potential errors during file reading
        logging.error(f"Error reading file '{file_path}': {e}")
        return None

def split_text_into_paragraphs(text: str) -> list[str]:
    """
    Splits given text into paragraphs based on one or more newline characters
    potentially separated by whitespace.

    Removes empty paragraphs or paragraphs containing only whitespace.

    Args:
        text: The input string containing the text to be split.

    Returns:
        A list of strings, where each string is a non-empty paragraph.
        Returns an empty list if the input text is None or empty.
    """
    if not text:
        return []

    # Split by one or more newlines, potentially with whitespace lines in between
    # This handles \n\n, \n \n, \n\t\n etc.
    paragraphs = re.split(r'\n\s*\n', text.strip())

    # Filter out empty strings or strings containing only whitespace after stripping
    non_empty_paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return non_empty_paragraphs

# --- New function to split paragraphs into sentences --- (Added function)
def _split_paragraphs_into_sentences(paragraphs: List[str]) -> List[str]:
    """Splits a list of paragraphs into a list of sentences using NLTK.
    Also removes summary markers like 'Resumo X' from the beginning of sentences.
    """
    all_sentences = []
    if not paragraphs:
        return []
    
    # Regex to find markers like "Resumo 1", "Resumo 2:", etc. at the start of a string
    # It looks for "Resumo" (case-insensitive) followed by a number, optionally a colon, and whitespace.
    summary_marker_regex = re.compile(r"^resumo\s*\d+\s*:?\s*", re.IGNORECASE)

    for paragraph in paragraphs:
        if paragraph: # Ensure paragraph is not empty
            try:
                sentences_in_paragraph = nltk.sent_tokenize(paragraph, language='portuguese')
                cleaned_sentences = []
                for sent in sentences_in_paragraph:
                    # Remove the marker from the beginning of the sentence if present
                    cleaned_sent = summary_marker_regex.sub("", sent).strip()
                    if cleaned_sent: # Add only if not empty after cleaning
                        cleaned_sentences.append(cleaned_sent)
                all_sentences.extend(cleaned_sentences)
            except Exception as e:
                logging.warning(f"NLTK sent_tokenize failed for paragraph '{paragraph[:50]}...': {e}. Treating paragraph as a single sentence.")
                # Fallback: treat the whole paragraph as a single sentence if tokenization fails
                # Still try to clean it
                cleaned_paragraph = summary_marker_regex.sub("", paragraph).strip()
                if cleaned_paragraph: # Add only if non-empty after cleaning
                    all_sentences.append(cleaned_paragraph)
    return all_sentences

def load_and_prepare_data(acordao_path: str, resumo_path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[None, None]:
    """
    Loads the main document (acordao) and summary document (resumo),
    splits them into paragraphs/claims, and structures them with metadata.
    Includes page number in metadata for PDF acórdãos.
    Resumo is now split into sentences, each treated as a claim.

    Args:
        acordao_path: Path to the main document file (.txt or .pdf).
        resumo_path: Path to summary file (.txt assumed, can be .pdf).

    Returns:
        A tuple containing two lists:
        1. acordao_chunks: List of dictionaries, each representing a paragraph
           chunk from main document with its text and metadata (including page_number for PDFs).
        2. resumo_claims: List of dictionaries, each representing a claim
           (now a sentence) from summary with its text and metadata.
        Returns (None, None) if either document cannot be loaded successfully.
    """
    logging.info(f"Starting data loading and preparation for:")
    logging.info(f"  Acordão: {acordao_path}")
    logging.info(f"  Resumo: {resumo_path}")

    # Load documents
    acordao_content = load_document(acordao_path)
    if acordao_content is None:
        logging.error(f"Failed to load the main document (acórdão) from: {acordao_path}")
        return None, None

    # Process Acordão based on its type (str for TXT, list for PDF)
    acordao_chunks_structured: List[Dict[str, Any]] = []
    global_chunk_index = 0

    if isinstance(acordao_content, str): # Handle TXT Acordão
        logging.info("Processing Acordão as TXT file.")
        acordao_paragraphs = split_text_into_paragraphs(acordao_content)
        for paragraph_text in acordao_paragraphs:
            metadata = {
                "source": acordao_path,
                "chunk_index": global_chunk_index,
                "chunk_type": "paragraph"
            }
            acordao_chunks_structured.append({
                "text": paragraph_text,
                "metadata": metadata
            })
            global_chunk_index += 1

    elif isinstance(acordao_content, list): # Handle PDF Acordão (list of tuples)
        logging.info("Processing Acordão as PDF file.")
        for page_num, page_text in acordao_content:
            paragraphs_on_page = split_text_into_paragraphs(page_text)
            for paragraph_text in paragraphs_on_page:
                metadata = {
                    "source": acordao_path,
                    "chunk_index": global_chunk_index,
                    "chunk_type": "paragraph",
                    "page_number": page_num  # Add page number here
                }
                acordao_chunks_structured.append({
                    "text": paragraph_text,
                    "metadata": metadata
                })
                global_chunk_index += 1
    else:
         # Should not happen based on load_document's return types, but checking defensively
         logging.error(f"Unexpected content type returned from load_document for Acordão: {type(acordao_content)}")
         return None, None

    logging.info(f"Processed {len(acordao_chunks_structured)} chunks from the acórdão.")

    # Process Resumo (only if a path provided)
    resumo_claims_structured: List[Dict[str, Any]] = []
    if resumo_path and os.path.exists(resumo_path):
        logging.info(f"Loading and processing resumo from: {resumo_path}")
        resumo_text_content = load_document(resumo_path) # Renamed variable for clarity
        if resumo_text_content is not None and isinstance(resumo_text_content, str): # Check if load successful as string
            # First, split the whole resumo text into paragraphs
            resumo_paragraphs = split_text_into_paragraphs(resumo_text_content)
            # Then, split each paragraph into sentences to get claims
            resumo_sentences = _split_paragraphs_into_sentences(resumo_paragraphs) # MODIFIED LOGIC

            for i, claim_text in enumerate(resumo_sentences):
                metadata = {
                    "source": resumo_path,
                    "claim_index": i, # Index now refers to sentence index across all paragraphs
                    "claim_type": "sentence" # Added type for clarity
                }
                resumo_claims_structured.append({
                    "text": claim_text,
                    "metadata": metadata
                })
            logging.info(f"Processed {len(resumo_claims_structured)} claims (sentences) from the resumo.")
        else:
            # Log warning if resumo path provided but load failed or not string
            logging.warning(f"Failed to load or process the provided resumo file as text: {resumo_path}")
            # Optionally, you might want to return None, None here if resumo is mandatory
            # For now, allow indexing to proceed without resumo if it fails loading
    elif resumo_path:
        # Log warning if path was provided but doesnt exist
        logging.warning(f"Resumo path provided, but file not found: {resumo_path}")
    else:
        logging.info("No resumo path provided, skipping resumo processing.")

    return acordao_chunks_structured, resumo_claims_structured

def pdf_to_docs(pdf_path: str) -> List[Dict]:
    """
    Reads a PDF, splits it into paragraphs, generates normalized embeddings,
    and returns a list of dictionaries suitable for indexing.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dictionaries, each with {'id', 'processo', 'text', 'vector'}.
        Returns an empty list if the PDF cannot be read or processed.
    """
    docs = []
    try:
        logging.info(f"Reading PDF: {pdf_path}")
        reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        logging.info(f"PDF has {num_pages} pages.")

        paragraphs = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Simple split by double newline, then filter empty strings
                    page_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', page_text) if p.strip()]
                    # Alternative: more robust paragraph splitting if needed
                    paragraphs.extend([(i + 1, para) for para in page_paragraphs]) # Store page number too
            except Exception as page_err:
                logging.warning(f"Could not extract text from page {i+1} of {pdf_path}: {page_err}")
                continue # Skip page on error

        if not paragraphs:
            logging.warning(f"No text paragraphs extracted from PDF: {pdf_path}")
            return []

        logging.info(f"Extracted {len(paragraphs)} paragraphs from PDF.")

        # Extract 'processo' NNN-YYYY from filename
        filename = os.path.basename(pdf_path)
        processo_id = _extract_processo_from_filename(filename)
        if not processo_id:
            logging.error(f"Failed to determine 'processo' ID for {pdf_path}. Cannot proceed with embedding.")
            return [] # Cannot index without processo_id

        logging.info(f"Generating embeddings for {len(paragraphs)} paragraphs...")
        embedder = get_embedding_model()
        paragraph_texts = [text for _, text in paragraphs]
        
        # Generate embeddings with normalization
        vectors = embedder.encode(
            paragraph_texts, 
            batch_size=32, # Adjust batch size based on VRAM
            show_progress_bar=True, 
            normalize_embeddings=True # Normalize embeddings as requested
        )
        logging.info("Embeddings generated.")

        # Prepare documents for indexing
        for idx, ((page_num, text), vector) in enumerate(zip(paragraphs, vectors)):
            doc_id = f"{processo_id}_p{page_num}_idx{idx}" # Unique ID per paragraph
            docs.append({
                "id": doc_id,
                "processo": processo_id, # Partition key
                "text": text,
                "vector": vector.tolist() # Convert numpy array to list for JSON serialization
            })

        logging.info(f"Prepared {len(docs)} documents with embeddings for {processo_id}.")
        return docs

    except FileNotFoundError:
        logging.error(f"PDF file not found: {pdf_path}")
        return []
    except Exception as e:
        logging.error(f"Failed to process PDF {pdf_path}: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    # This block executes only when script run directly
    # (e.g., python src/data_loader.py from project root)
    # Useful for basic testing of module's main function.

    print("\n" + "="*20 + " Running data_loader.py Direct Execution Test " + "="*20)

    # Define paths relative to project root.
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_path) # Go up one level from src/
    data_folder = os.path.join(project_root, 'data')

    print(f"Project Root detected: {project_root}")
    print(f"Data folder path: {data_folder}")

    # Select files from data folder
    # Using Acórdão 733 as an example
    acordao_filename = "Acórdão 733 de 2025 Plenário.pdf"
    resumo_filename = "Acórdão 733-2025 resumos.txt"
    # You can change these filenames to test other pairs

    acordao_file_path = os.path.join(data_folder, acordao_filename)
    resumo_file_path = os.path.join(data_folder, resumo_filename)

    print(f"\nAttempting to load and prepare data for:")
    print(f"  Acordão: {acordao_file_path}")
    print(f"  Resumo:  {resumo_file_path}")

    # Check if files exist before calling
    acordao_exists = os.path.exists(acordao_file_path)
    resumo_exists = os.path.exists(resumo_file_path)

    if not acordao_exists:
        print(f"ERROR: Acordão file not found: {acordao_file_path}")
    if not resumo_exists:
        print(f"ERROR: Resumo file not found: {resumo_file_path}")

    if acordao_exists and resumo_exists:
        # Call the main function
        acordao_chunks, resumo_claims = load_and_prepare_data(acordao_file_path, resumo_file_path)

        # Print results summary
        if acordao_chunks is not None and resumo_claims is not None:
            print("\n--- Load and Prepare Results ---")
            print(f"Number of Acordão Chunks generated: {len(acordao_chunks)}")
            print(f"Number of Resumo Claims (Sentences) generated:  {len(resumo_claims)}") # Updated label

            if acordao_chunks:
                print("\nFirst Acordão Chunk (first 150 chars):")
                print(f"  Text: '{acordao_chunks[0]['text'][:150].replace(chr(10), ' ')}...'") # Show preview
                print(f"  Metadata: {acordao_chunks[0]['metadata']}") # Should include page_number now
                # Print last chunk metadata too, see page number progression
                if len(acordao_chunks) > 1:
                     print("\nLast Acordão Chunk Metadata:")
                     print(f"  Metadata: {acordao_chunks[-1]['metadata']}")
            else:
                print("\nWarning: No Acordão chunks were generated.")

            if resumo_claims:
                print("\nFirst Resumo Claim:")
                print(f"  Text: '{resumo_claims[0]['text']}'") # Usually short enough to show full
                print(f"  Metadata: {resumo_claims[0]['metadata']}")
            else:
                print("\nWarning: No Resumo claims were generated.")
        else:
            print("\n--- Load and Prepare FAILED ---")
            print("  load_and_prepare_data returned None, None. Check logs for errors.")

    print("\n" + "="*20 + " End of data_loader.py Direct Execution Test " + "="*20)
