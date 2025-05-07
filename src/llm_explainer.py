import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - EXPLAINER - %(levelname)s - %(message)s')

# --- Constants --- 
LLM_EXPLAINER_MODEL_ID = "cnmoro/Mistral-7B-Portuguese-Instruct" # Mistral-7B-Instruct-v0.1 based
# Ensure this matches your available hardware and desired precision
# For BF16, CUDA compute capability >= 8.0 (Ampere) is typically needed.
# If not available, it might fall back or you might need torch.float16.
MODEL_TORCH_DTYPE = torch.bfloat16 

# --- Cached Model and Tokenizer --- 
_explainer_model_instance: Optional[AutoModelForCausalLM] = None
_explainer_tokenizer_instance: Optional[AutoTokenizer] = None

def load_explainer_model_and_tokenizer(
    model_id: str = LLM_EXPLAINER_MODEL_ID
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads and caches the LLM model (quantized if applicable) and tokenizer for explanation.
    Uses BitsAndBytesConfig for 4-bit quantization if CUDA is available.
    """
    global _explainer_model_instance, _explainer_tokenizer_instance

    if _explainer_model_instance is not None and _explainer_tokenizer_instance is not None:
        logging.debug("Returning cached explainer LLM model and tokenizer.")
        return _explainer_model_instance, _explainer_tokenizer_instance

    logging.info(f"Loading explainer LLM model and tokenizer for: {model_id}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Configure 4-bit quantization (optional, adjust as needed)
        quantization_config = None
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=MODEL_TORCH_DTYPE, # BF16 or FP16
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logging.info("Using 4-bit quantization for explainer model.")
        else:
            logging.info("CUDA not available, loading explainer model in full precision on CPU (may be slow).")

        model_kwargs = {
            "torch_dtype": MODEL_TORCH_DTYPE if device == 'cuda' else torch.float32,
            "low_cpu_mem_usage": True, # Useful for large models
            # "trust_remote_code": True, # if model requires it
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto" # Handles multi-GPU and CPU offloading
        # else, model will be loaded on CPU first, then moved if device='cuda' is specified
        
        logging.info(f"Loading model '{model_id}' with kwargs: {model_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        if not quantization_config and device == 'cuda': # If not quantized, move to GPU if available
             model = model.to(device)

        logging.info(f"Explainer model '{model_id}' loaded on device: {model.device if hasattr(model, 'device') else device}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # For decoder-only models, padding side is often right, but check model specifics
            # tokenizer.padding_side = "left" # For Gemma
            tokenizer.padding_side = "right" # Mistral typically uses right padding
            logging.info(f"Set tokenizer pad_token to eos_token and padding_side to '{tokenizer.padding_side}'.")
        
        _explainer_model_instance = model
        _explainer_tokenizer_instance = tokenizer
        _explainer_model_instance.eval()
        
        return _explainer_model_instance, _explainer_tokenizer_instance

    except Exception as e:
        logging.error(f"Failed to load explainer model or tokenizer '{model_id}': {e}", exc_info=True)
        raise RuntimeError(f"Explainer model/tokenizer load failed: {e}") from e

def _format_explanation_prompt(query: str, contexts: List[str]) -> str:
    """
    Formats the prompt for the Mistral-Instruct model to generate an explanation.
    The prompt instructs the model to explain if paragraphs confirm the affirmation,
    and *then* state CONFERE/NÃO CONFERE. The `explain` function should extract the rationale.
    """
    context_str = "\n\n".join(f"Parágrafo {i+1}:\n{chunk}" for i, chunk in enumerate(contexts) if chunk.strip())
    if not context_str:
        context_str = "Nenhum parágrafo de contexto foi fornecido."

    # Mistral Instruct template: <s>[INST] User Query [/INST] Model Answer</s>
    # The user query is the full instruction set.
    prompt = f"<s>[INST] Explique em até 3 frases se os parágrafos abaixo CONFIRMAM a afirmação e, em seguida, responda apenas CONFERE ou NÃO CONFERE.\n\nAfirmação: {query}\n\nParágrafos:\n{context_str}\n\n[/INST]"
    return prompt

def explain(query: str, list_of_texts: List[str], max_new_tokens: int = 150, temperature: float = 0.1) -> str:
    """
    Generates a short rationale explaining whether the provided texts support the query.
    Uses the Mistral-7B-Portuguese-Instruct model.

    Args:
        query: The query or claim string.
        list_of_texts: A list of text strings (e.g., top N reranked document chunks).
        max_new_tokens: Max new tokens for the LLM to generate for the explanation.
        temperature: Temperature for LLM generation (lower for more factual).

    Returns:
        A string containing the generated rationale (explanation part).
        Returns a default error message if generation fails or output is unexpected.
    """
    if not query or not list_of_texts:
        logging.warning("Query and list_of_texts must be provided for explanation.")
        return "Não foi possível gerar a explicação: dados de entrada ausentes."

    try:
        model, tokenizer = load_explainer_model_and_tokenizer()
        device = model.device if hasattr(model, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

        prompt_text = _format_explanation_prompt(query, list_of_texts)
        logging.debug(f"Generated explanation prompt:\n{prompt_text}")

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=False).to(device)
        # Ensure input_ids does not exceed model max length (e.g., 4096 for some Mistral variants)
        # This check is more for the prompt itself; max_new_tokens controls output length.
        # if inputs.input_ids.shape[1] > tokenizer.model_max_length:
        #     logging.warning(f"Prompt length ({inputs.input_ids.shape[1]}) exceeds model max length ({tokenizer.model_max_length}). Truncation might occur unexpectedly.")

        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9, # Sample from the smallest set of tokens whose cumulative probability exceeds top_p
            "do_sample": True, # Required for temperature and top_p to have effect
            "pad_token_id": tokenizer.eos_token_id, # Important for open-ended generation
            "eos_token_id": tokenizer.eos_token_id
        }
        
        logging.info(f"Generating explanation with {LLM_EXPLAINER_MODEL_ID}...")
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        
        # Decode the generated tokens, skipping special tokens and the prompt part
        # The output_ids will contain the prompt + generated text.
        # We need to slice off the prompt part from the output.
        prompt_length = inputs.input_ids.shape[1]
        generated_text_ids = output_ids[0, prompt_length:]
        llm_response_text = tokenizer.decode(generated_text_ids, skip_special_tokens=True).strip()
        
        logging.info(f"LLM Raw Response for explanation: {llm_response_text}")

        # The prompt asks for: "Explique... e, em seguida, responda apenas CONFERE ou NÃO CONFERE."
        # We want to extract the explanation part before the final CONFERE/NÃO CONFERE.
        # This can be tricky. A simple approach is to look for these keywords.
        match_confere = re.search(r"(CONFERE|NÃO\s*CONFERE)$", llm_response_text, re.IGNORECASE | re.MULTILINE)
        rationale = llm_response_text
        if match_confere:
            rationale = llm_response_text[:match_confere.start()].strip()
            # If rationale is empty after stripping, it means LLM only gave CONFERE/NAO CONFERE
            if not rationale:
                rationale = f"O modelo indicou '{match_confere.group(1)}' mas não forneceu uma explicação textual detalhada antes disso." 
        else:
            # If the keywords are not found at the end, the whole response might be the rationale
            # or the LLM didn't follow the format. We'll take the whole response as is.
            logging.warning("LLM did not strictly follow the CONFERE/NÃO CONFERE ending format. Using full response as rationale.")
        
        if not rationale.strip(): # Handle cases where rationale might become empty
            rationale = "A explicação gerada pelo LLM estava vazia ou não seguiu o formato esperado."
            
        return rationale

    except Exception as e:
        logging.error(f"Failed to generate explanation for query '{query[:50]}...': {e}", exc_info=True)
        return f"Erro ao gerar explicação: {str(e)}"

# Example (for testing if run directly)
if __name__ == '__main__':
    import re # Needed for the main block too
    test_query = "O tribunal superior confirmou a decisão da instância inferior."
    test_contexts = [
        "Trecho A: A corte de apelação manteve a sentença original em todos os seus termos.",
        "Trecho B: A decisão foi proferida pelo juiz de primeiro grau e não houve recurso.",
        "Trecho C: O recurso especial não foi conhecido pelo STJ, por ausência de prequestionamento."
    ]

    print(f"Test Case - Query: {test_query}")
    print(f"          Contexts: {test_contexts}\n")
    
    # First run will download and cache the model (can take time and VRAM)
    try:
        explanation = explain(test_query, test_contexts)
        print(f"Generated Explanation:\n---\n{explanation}\n---")
        
        # Test with empty context
        explanation_empty_ctx = explain(test_query, [])
        print(f"\nGenerated Explanation (Empty Context):\n---\n{explanation_empty_ctx}\n---")

        # Test with more complex context
        test_query_2 = "A empresa pública deve seguir o teto remuneratório."
        test_contexts_2 = [
            "Conforme a Constituição, o teto se aplica a todos os entes da administração direta e indireta.",
            "As empresas estatais, contudo, possuem regime jurídico próprio, conforme Lei X.",
            "Decisão do STF no RE YYYY estabeleceu que empresas públicas que não recebem recursos da União para custeio não se submetem ao teto."
        ]
        print(f"\nTest Case 2 - Query: {test_query_2}")
        print(f"          Contexts: {test_contexts_2}\n")
        explanation_2 = explain(test_query_2, test_contexts_2)
        print(f"Generated Explanation:\n---\n{explanation_2}\n---")

    except RuntimeError as e:
        print(f"RUNTIME ERROR during llm_explainer.py test: {e}")
        print("This often indicates an issue with model loading (e.g., OOM, HuggingFace auth, or network).")
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")

    logging.info("LLM Explainer direct execution test finished.") 