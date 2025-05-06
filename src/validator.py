import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any, Optional, Tuple
import re # Import regular expressions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - VALIDATOR - %(levelname)s - %(message)s')

# --- Constants ---
LLM_MODEL_ID = "google/gemma-2b-it"

# --- Cached Model and Tokenizer ---
_llm_model_instance: Optional[AutoModelForCausalLM] = None
_llm_tokenizer_instance: Optional[AutoTokenizer] = None

def load_llm_model_and_tokenizer(
    model_id: str = LLM_MODEL_ID
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads and caches the LLM model (quantized) and tokenizer.

    Args:
        model_id: The Hugging Face ID of model to load.

    Returns:
        A tuple containing the loaded model and tokenizer.

    Raises:
        RuntimeError: If the model or tokenizer fails to load.
        ImportError: If required libraries (transformers, torch, accelerate, bitsandbytes) not installed.
    """
    global _llm_model_instance, _llm_tokenizer_instance

    if _llm_model_instance is not None and _llm_tokenizer_instance is not None:
        logging.debug("Returning cached LLM model and tokenizer.")
        return _llm_model_instance, _llm_tokenizer_instance

    logging.info(f"Loading LLM model and tokenizer for: {model_id}")
    try:
        # Configure 4-bit quantization (should work for Gemma)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # Use torch.float16 if bfloat16 not supported
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        logging.info("Loading model with 4-bit quantization...")
        if not torch.cuda.is_available():
             raise RuntimeError("CUDA (GPU) not available. Quantized model loading requires GPU.")

        # Gemma doesn't need trust_remote_code or specific attn_implementation typically
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype="auto", # Use torch.float16 if bfloat16 cause issues
            low_cpu_mem_usage=True,
        )
        logging.info("Model loaded successfully.")

        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Gemma might not need explicit pad token setting, but good practice to check
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left" # Or "right", depending on model training
            logging.info("Set tokenizer pad_token to eos_token and padding_side to left.")
        logging.info("Tokenizer loaded successfully.")

        # Cache the instances
        _llm_model_instance = model
        _llm_tokenizer_instance = tokenizer

        # Log memory usage (optional)
        try:
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
            logging.info(f"GPU Memory after load: Allocated={allocated_gb:.2f} GB, Reserved={reserved_gb:.2f} GB")
        except Exception:
            logging.warning("Could not get GPU memory usage.", exc_info=True)


        return _llm_model_instance, _llm_tokenizer_instance

    except ImportError as e:
         logging.error(f"ImportError: Required library not found. Install transformers, torch, accelerate, bitsandbytes. Details: {e}")
         raise
    except Exception as e:
        logging.error(f"Failed to load LLM model or tokenizer: {e}", exc_info=True)
        _llm_model_instance = None # Ensure cache is clear on failure
        _llm_tokenizer_instance = None
        raise RuntimeError(f"Failed to load LLM model or tokenizer: {e}") from e


def _format_validation_prompt(claim: str, context_chunks: List[str]) -> str:
    """
    Formats prompt for Gemma Instruct model to validate claim based on context.

    Args:
        claim: The claim text from the summary.
        context_chunks: A list of relevant text chunks retrieved from original document.

    Returns:
        The formatted prompt string.
    """
    context_str = "\\n\\n".join(f"Trecho {i+1}:\\n{chunk}" for i, chunk in enumerate(context_chunks))

    user_message = f"""Você é um assistente de IA altamente preciso e literal, especializado em validar afirmações em documentos jurídicos do TCU. Sua tarefa é determinar se uma 'Alegação' é 'Correta' ou 'Incorreta' baseando-se *única, exclusiva e estritamente* nas informações contidas nos 'Trechos do Documento Original' fornecidos.

**REGRAS ABSOLUTAS (Siga-as Implacavelmente):**
1.  **NÃO USE CONHECIMENTO EXTERNO.** Sua análise deve ser 100% baseada nos trechos fornecidos.
2.  **NÃO FAÇA INFERÊNCIAS OU SUPOSIÇÕES.** Se a informação não está explicitamente declarada nos trechos, ela não existe para esta tarefa.
3.  **SEJA LITERAL.** Compare a alegação com os trechos palavra por palavra, significado por significado.
4.  **PARA SER 'Correta',** TODAS as partes da alegação devem ser explicitamente confirmadas por um ou mais trechos. Se qualquer parte da alegação não for explicitamente confirmada, ela é 'Incorreta'.
5.  **PARA SER 'Incorreta',** UMA OU MAIS das seguintes condições devem ser verdadeiras:
    a.  Um ou mais trechos contradizem diretamente a alegação.
    b.  Os trechos fornecidos NÃO CONTÊM informação suficiente para confirmar a alegação (mesmo que não a contradigam diretamente).

**PROCESSO DE ANÁLISE INTERNO (Use para guiar seu raciocínio, NÃO inclua no output final - apenas pense nesses passos):**
    A. Leia a 'Alegação' cuidadosamente. Identifique cada fato ou afirmação individual nela contida.
    B. Para CADA fato/afirmação na alegação:
        i.  Examine TODOS os 'Trechos do Documento Original'.
        ii. Procure por uma declaração explícita que CONFIRME este fato/afirmação.
        iii.Procure por uma declaração explícita que CONTRADIGA este fato/afirmação.
    C. Avaliação Final (baseada no processo interno):
        i.  Se TODOS os fatos/afirmações da alegação foram explicitamente CONFIRMADOS pelos trechos E NENHUM foi contradito, a alegação é 'Correta'.
        ii. Se QUALQUER fato/afirmação da alegação foi explicitamente CONTRADITO pelos trechos, a alegação é 'Incorreta'.
        iii.Se QUALQUER fato/afirmação da alegação NÃO PODE SER CONFIRMADO (ou seja, a informação está AUSENTE nos trechos), a alegação é 'Incorreta'.

**Trechos do Documento Original:**
{context_str}

**Alegação a ser validada:**
{claim}

**FORMATO DE SAÍDA OBRIGATÓRIO E FINAL (NÃO inclua NADA MAIS em sua resposta, apenas estas duas linhas):**
Resultado: [Correta/Incorreta]
Justificativa: [Se 'Correta': "A alegação é confirmada pelo(s) Trecho(s) X, Y, que afirmam [citação relevante ou paráfrase muito próxima]." OU Se 'Incorreta' por contradição: "A alegação é contradita pelo(s) Trecho(s) X, Y, que afirmam [citação relevante ou paráfrase]." OU Se 'Incorreta' por ausência de informação: "A informação necessária para confirmar '[parte específica da alegação]' não foi encontrada nos trechos fornecidos."
]"""

    # Combine into final prompt format for Gemma
    prompt = f"<start_of_turn>user\\n{user_message.strip()}<end_of_turn>\\n<start_of_turn>model\\n"

    return prompt


def validate_claim_with_llm(
    query_claim: str,
    retrieved_chunks: Dict[str, Any],
    max_new_tokens: int = 150, # Increased from 100
    temperature: float = 0.1, # Lower temperature for more deterministic output
) -> Optional[Dict[str, Any]]:
    """
    Validates claim using LLM based on retrieved context chunks.

    Args:
        query_claim: The claim text from the summary.
        retrieved_chunks: Dictionary returned by retrieve_relevant_chunks,
                          containing 'documents', 'metadatas', etc.
        max_new_tokens: Maximum number of tokens for LLM to generate.
        temperature: Sampling temperature for generation. Lower is more focused.

    Returns:
        A dictionary with 'Resultado', 'Justificativa', 'input_tokens', 'output_tokens'
        keys, or None if validation fails or context is missing.
    """
    if not query_claim:
        logging.warning("Received empty query claim. Cannot validate.")
        return None
    if not retrieved_chunks or not retrieved_chunks.get('documents') or not retrieved_chunks['documents'][0]:
        logging.warning("No retrieved documents found in input. Cannot validate.")
        return None

    # Extract the actual text content from possibly nested list
    context_documents = retrieved_chunks['documents'][0]

    try:
        # Load Model and Tokenizer (uses cache)
        model, tokenizer = load_llm_model_and_tokenizer()

        # Format the Prompt
        prompt = _format_validation_prompt(query_claim, context_documents)
        logging.debug(f"Formatted prompt for LLM:\n{prompt}")

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).to(model.device)
        input_token_count = inputs['input_ids'].shape[1]
        logging.debug(f"Input tokens: {input_token_count}")

        # Generate the response
        logging.info(f"Generating validation for claim: '{query_claim[:100]}...'")
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0, # Only sample if temperature not 0
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }

        # Ensure pad_token_id set if tokenizer doesnt have one naturally
        if generation_args["pad_token_id"] is None:
             generation_args["pad_token_id"] = generation_args["eos_token_id"]


        with torch.no_grad(): # Inference dont need gradient calculation
            outputs = model.generate(**inputs, **generation_args)

        # Decode the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        llm_response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        output_token_count = len(generated_ids)
        logging.debug(f"Output tokens: {output_token_count}")

        logging.info(f"LLM Raw Response: {llm_response_text}")

        # Parse the response (Revised Logic)
        result_dict = {
            "Resultado": "Erro",
            "Justificativa": "Falha ao parsear resposta do LLM.",
            "input_tokens": input_token_count,
            "output_tokens": output_token_count
        }

        # Use regex to find Resultado, ignore markdown and case for Correta/Incorreta
        # Make regex more robust: handles optional bolding of "Resultado", more whitespace, case-insensitive "Resultado", multiline
        match_resultado = re.search(r"^\s*\*?\s*Resultado\s*:\s*\*?\s*([Cc]orreta|[Ii]ncorreta)\*?", 
                                    llm_response_text, 
                                    re.IGNORECASE | re.MULTILINE)

        if match_resultado:
            # Normalize to capitalize first letter only (e.g., "Correta" or "Incorreta")
            parsed_status = match_resultado.group(1).strip().capitalize()
            if parsed_status in ["Correta", "Incorreta"]:
                result_dict["Resultado"] = parsed_status
                logging.info(f"Extracted Resultado: {parsed_status}")

                # Extract justification from text *after* matched "Resultado" line
                remaining_text = llm_response_text[match_resultado.end():].strip()

                # Remove potential leading "Justificativa:" prefix if present (case-insensitive)
                if remaining_text.lower().startswith("justificativa:"):
                    remaining_text = remaining_text.split(":", 1)[1].strip()

                # Assign justification based on status
                if result_dict["Resultado"] == "Correta":
                    # For Correta, justification is optional or N/A
                    result_dict["Justificativa"] = remaining_text if remaining_text else "N/A"
                else: # If Incorreta, justification expected
                    result_dict["Justificativa"] = remaining_text if remaining_text else "Justificativa não fornecida pelo LLM."
            else:
                # Pattern matched but value unexpected
                logging.warning(f"Regex matched 'Resultado:' but found unexpected value: '{match_resultado.group(1)}'")
                result_dict["Justificativa"] = f"Formato inesperado (valor Resultado): {llm_response_text}"
        else:
            # Pattern "Resultado:..." not found
            logging.warning(f"Could not find 'Resultado:' pattern in LLM response: '{llm_response_text}'")
            result_dict["Justificativa"] = f"Formato inesperado (sem Resultado): {llm_response_text}"

        logging.info(f"Parsed Result: {result_dict['Resultado']}")
        return result_dict

    except RuntimeError as e:
         logging.error(f"Runtime error during LLM validation (possibly OOM or model loading failed): {e}")
         return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during LLM validation: {e}", exc_info=True)
        return None


# --- Example Usage Block (for direct testing) ---
if __name__ == "__main__":
    print("\n" + "="*20 + " Running validator.py Direct Execution Test " + "="*20)

    # Dummy Data (Simulating retriever output)
    # In real scenario, this would come from retrieve_relevant_chunks
    sample_claim = "O BNDES é uma estatal dependente da União"
    sample_context = {
        'ids': [['doc_id_1', 'doc_id_2']],
        'documents': [[
            "Trecho 1: O Banco Nacional de Desenvolvimento Econômico e Social (BNDES) é uma empresa pública federal...",
            "Trecho 2: Conforme análise do Ministério da Fazenda, o BNDES se enquadra como empresa estatal dependente, sujeita ao teto remuneratório."
        ]],
        'metadatas': [[{'source': 'doc.pdf', 'page': 1}, {'source': 'doc.pdf', 'page': 5}]],
        'distances': [[0.1, 0.2]]
    }

    print(f'Test Claim: "{sample_claim}"')
    print(f"Context Chunks: {sample_context['documents'][0]}")
    print("-" * 20)

    validation_result = None
    try:
        # Run Validation
        # First time this runs, it will load model (can take time and VRAM)
        validation_result = validate_claim_with_llm(sample_claim, sample_context)

    except RuntimeError as e:
         print(f"RUNTIME ERROR during test: {e}")
         print("This might be an OOM error or model loading failure. Check VRAM and logs.")
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")


    # Print Result
    print("-" * 20)
    if validation_result:
        print("--- Validation Result ---")
        print(f"Resultado: {validation_result.get('Resultado', 'N/A')}")
        print(f"Justificativa: {validation_result.get('Justificativa', 'N/A')}")
    else:
        print("--- Validation FAILED ---")
        print("Check logs above for errors (model loading, OOM, etc.)")


    print("\n" + "="*20 + " End of validator.py Direct Execution Test " + "="*20) 