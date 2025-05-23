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
    Formats prompt for Mixtral Instruct model to validate claim based on context.

    Args:
        claim: The claim text from the summary.
        context_chunks: A list of relevant text chunks retrieved from original document.

    Returns:
        The formatted prompt string.
    """
    context_str = "\\\\n\\\\n".join(f"Trecho {i+1}:\\\\n{chunk}" for i, chunk in enumerate(context_chunks))

    user_content = f\"\"\"Você é um assistente de IA especialista em análise de documentos jurídicos do TCU. Sua tarefa é avaliar se a 'Alegação' fornecida é 'Correta' ou 'Incorreta' com base *apenas e estritamente* nos 'Trechos do Documento Original' que acompanham a alegação.

**Instruções Gerais:**
1.  **Foco no Contexto:** Sua decisão deve ser 100% baseada nos 'Trechos do Documento Original'. Não utilize conhecimento externo ou informações que não estejam presentes nesses trechos.
2.  **Sem Suposições:** Não faça inferências que vão além do que está explicitamente declarado ou fortemente implicado pelos trechos.
3.  **Essência da Alegação:** Concentre-se em determinar se a afirmação principal ou a essência da 'Alegação' é validada pelos trechos.

**Como Avaliar e Responder:**

*   **Para classificar como 'Correta':**
    *   A afirmação principal da 'Alegação' deve ser claramente suportada pelos 'Trechos do Documento Original'.
    *   Não deve haver contradições diretas entre a 'Alegação' e os 'Trechos do Documento Original' em relação aos seus pontos chave.
    *   Pequenas variações de redação entre a alegação e o texto de suporte são aceitáveis, desde que o significado central seja o mesmo.
    *   **Justificativa para 'Correta':** Mencione qual(is) trecho(s) suporta(m) a alegação e explique brevemente como. Ex: "O Trecho X confirma que [aspecto da alegação]."

*   **Para classificar como 'Incorreta':**
    *   Um ou mais pontos chave da 'Alegação' são diretamente contraditos por informações nos 'Trechos do Documento Original'.
    *   OU, os 'Trechos do Documento Original' não fornecem informações suficientes para confirmar a afirmação principal da 'Alegação'.
    *   **Justificativa para 'Incorreta' (Contradição):** Mencione qual(is) trecho(s) contradiz(em) a alegação e explique brevemente. Ex: "O Trecho Y contradiz a alegação ao afirmar que [informação contrastante]."
    *   **Justificativa para 'Incorreta' (Ausência de Informação):** Indique que a informação essencial para validar a alegação não foi encontrada nos trechos. Ex: "Os trechos fornecidos não contêm informação sobre [aspecto específico da alegação não encontrado]."

**Exemplo de Alegação:**
"O contrato foi assinado em 2023 e previa a entrega de 50 computadores."

**Exemplos de Avaliação com Base em Trechos Hipotéticos:**
*   Se um trecho diz "O acordo celebrado em 2023 estipulava o fornecimento de cinquenta computadores.", a alegação seria 'Correta'.
*   Se um trecho diz "O contrato, firmado em 2022, previa a entrega de 50 computadores.", a alegação seria 'Incorreta' (contradição na data).
*   Se um trecho diz "O contrato previa a entrega de 50 computadores.", mas não menciona o ano, a alegação (considerando o ano como parte chave) seria 'Incorreta' (ausência de informação sobre o ano).
*   Se um trecho diz "O contrato foi assinado em 2023.", mas não menciona a quantidade de computadores, a alegação seria 'Incorreta' (ausência de informação sobre a quantidade).

**Trechos do Documento Original:**
{context_str}

**Alegação a ser validada:**
{claim}

**FORMATO DE SAÍDA OBRIGATÓRIO (Responda APENAS com as duas linhas abaixo, exatamente neste formato):**
Resultado: [Correta/Incorreta]
Justificativa: [Sua explicação concisa, seguindo as diretrizes acima.]\"\"\"

    # Mixtral Instruct format: <s> [INST] User Prompt [/INST] Model will generate from here.
    prompt = f"<s>[INST] {user_content.strip()} [/INST]"
    # The model should now generate "Resultado: ..." itself.
    return prompt


def validate_claim_with_llm(
    query_claim: str,
    retrieved_chunks: Dict[str, Any],
    max_new_tokens: int = 200,
    temperature: float = 0.01,
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

        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        llm_response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # LLM is now expected to generate "Resultado: ..." itself.
        # We no longer prepend "Resultado: " to llm_response_text.

        output_token_count = len(generated_ids)
        logging.debug(f"Output tokens: {output_token_count}")
        logging.info(f"LLM Raw Response: {llm_response_text}")

        result_dict = {
            "Resultado": "Erro",
            "Justificativa": "Falha ao parsear resposta do LLM.",
            "input_tokens": input_token_count,
            "output_tokens": output_token_count
        }

        # It looks for "Resultado" (case-insensitive) followed by ":"
        # then captures "Correta" or "Incorreta" (case-insensitive).
        match_resultado = re.search(
            r"^\s*Resultado\s*:\s*([Cc]orreta|[Ii]ncorreta)", # Greatly simplified regex
            llm_response_text,
            re.IGNORECASE | re.MULTILINE
        )

        if match_resultado:
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