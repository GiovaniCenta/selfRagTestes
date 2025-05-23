import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Optional, Tuple
import re # Added import for regex in explain function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - EXPLAINER - %(levelname)s - %(message)s')

# --- Constants --- 
# Updated Model ID to meta-llama/Llama-3.1-8B-Instruct
LLM_EXPLAINER_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
# This model should fit in BF16 on A100 GPUs.
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
        # --- REMOVING 8-BIT QUANTIZATION --- 
        quantization_config = None
        # if torch.cuda.is_available():
        #     quantization_config = BitsAndBytesConfig(
        #         load_in_8bit=True # Using 8-bit quantization for Llama 4 Scout
        #     )
        #     logging.info("Using 8-bit quantization for Llama 4 Scout explainer model.")
        # else:
        #     logging.warning("CUDA not available, loading Llama 4 Scout model in full precision on CPU. This WILL be extremely slow and likely fail due to memory.")
        logging.info(f"Attempting to load {LLM_EXPLAINER_MODEL_ID} in {MODEL_TORCH_DTYPE if device == 'cuda' else 'float32'} on {device}.")

        model_kwargs = {
            "low_cpu_mem_usage": True, 
            # "trust_remote_code": True, 
        }
        # --- Simplified loading logic: No quantization, direct dtype setting --- 
        if device == 'cuda':
             model_kwargs["torch_dtype"] = MODEL_TORCH_DTYPE
        else:
             # If no CUDA, use float32
             model_kwargs["torch_dtype"] = torch.float32
        
        logging.info(f"Loading model '{model_id}' with kwargs: {model_kwargs}")
        # Load model using AutoModelForCausalLM
        # If CUDA is available, transformers will often try to load directly to GPU with this setup.
        # If not, it loads on CPU.
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs) 
        
        # Explicitly move to GPU if loaded on CPU and CUDA is available
        if device == 'cuda' and model.device.type != 'cuda':
            logging.info(f"Moving model from {model.device} to {device}...")
            model = model.to(device)

        logging.info(f"Explainer model '{model_id}' loaded on device: {model.device if hasattr(model, 'device') else device}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # For decoder-only models, padding side is often right.
            # Llama 3 might have specific recommendations, but "right" is common.
            tokenizer.padding_side = "right" 
            logging.info(f"Set tokenizer pad_token to eos_token ({tokenizer.eos_token}) and padding_side to '{tokenizer.padding_side}'.")
        
        _explainer_model_instance = model
        _explainer_tokenizer_instance = tokenizer
        _explainer_model_instance.eval()
        
        return _explainer_model_instance, _explainer_tokenizer_instance

    except Exception as e:
        logging.error(f"Failed to load explainer model or tokenizer '{model_id}': {e}", exc_info=True)
        raise RuntimeError(f"Explainer model/tokenizer load failed: {e}") from e

def _format_explanation_prompt(query: str, contexts: List[str]) -> str:
    """
    Formats the prompt for the meta-llama/Llama-3.1-8B-Instruct model.
    Uses the official Llama 3.1 instruct format structure.
    """
    context_str = "\n\n".join(f"Parágrafo {i+1}:\n{chunk}" for i, chunk in enumerate(contexts) if chunk.strip())
    if not context_str:
        context_str = "Nenhum parágrafo de contexto foi fornecido."

    # System prompt with detailed instructions
    system_prompt_content = (
        f"Sua tarefa é analisar o conjunto de Parágrafos fornecidos no 'user prompt' e determinar se eles, como um todo, confirmam ou contradizem a Afirmação dada. Seja extremamente rigoroso, cético e analítico.\n\n"
        f"PASSO A PASSO DETALHADO PARA SUA ANÁLISE INTERNA:\n"
        f"1. DECOMPONHA A AFIRMAÇÃO: Identifique todas as suas partes ou sub-alegações constituintes. Cada uma delas deve ser verificada individualmente.\n"
        f"2. VERIFICAÇÃO RIGOROSA NOS PARÁGRAFOS: Para CADA sub-alegação da Afirmação, procure por evidência CLARA, DIRETA e INEQUÍVOCA nos Parágrafos que a confirme.\n"
        f"3. BUSCA POR CONTRADIÇÕES: Verifique também se há QUALQUER informação nos Parágrafos que CONTRADIGA explicitamente QUALQUER sub-alegação da Afirmação.\n\n"
        f"REGRAS ESTRITAS PARA O VEREDITO ÚNICO ('Correto' ou 'Incorreto'):\n"
        f"- Use 'Correto' SOMENTE E APENAS SE TODAS AS PARTES e sub-alegações da Afirmação forem CLARAMENTE e INEQUIVOCAMENTE confirmadas pelos Parágrafos E não houver NENHUMA contradição a QUALQUER parte da Afirmação.\n"
        f"- Use 'Incorreto' SE QUALQUER UMA DAS SEGUINTES CONDIÇÕES FOR VERDADEIRA: \n"
        f"    a) QUALQUER parte ou sub-alegação da Afirmação for explicitamente CONTRADITA por alguma informação nos Parágrafos; OU\n"
        f"    b) QUALQUER parte ou sub-alegação da Afirmação NÃO PUDER ser confirmada por falta de informação clara e direta nos Parágrafos (ou seja, a informação é ausente ou ambígua); OU\n"
        f"    c) Houver qualquer ambiguidade, dúvida razoável ou necessidade de inferência não explícita para conectar os Parágrafos à Afirmação.\n\n"
        f"SOBRE A JUSTIFICATIVA:\n"
        f"- Sua justificativa deve ser a consequência lógica direta da sua análise e da aplicação das regras de veredito. NÃO invente informações.\n"
        f"- NÃO forneça vereditos separados para cada parágrafo. O veredito é sobre a Afirmação como um todo, frente ao CONJUNTO de Parágrafos.\n\n"
        f"FORMATO OBRIGATÓRIO DA SUA RESPOSTA:\\n"
        f"Sua resposta DEVE OBRIGATORIAMENTE começar com a palavra 'Correto' ou 'Incorreto', seguida por um ponto final ('.') e uma nova linha.\\n"
        f"Exemplo de início: Correto.\\n"
        f"Exemplo de início: Incorreto.\\n"
        f"Logo APÓS o veredito (Correto. ou Incorreto.) E A NOVA LINHA, sua resposta DEVE continuar IMEDIATAMENTE com a frase 'Justificativa: ' (exatamente como escrito, incluindo o espaço após os dois pontos), seguida pela sua explicação. NÃO adicione linhas em branco extras ou qualquer outro texto entre a linha do veredito e a linha que começa com 'Justificativa: '.\\\\n"
        f"EXEMPLO COMPLETO DE RESPOSTA ESPERADA:\\\\n"
        f"Correto.\\\\n"
        f"Justificativa: [sua explicação aqui]\\\\n\\\\n"
        f"Ou\\\\n"
        f"Incorreto.\\\\n"
        f"Justificativa: [sua explicação aqui]\\\\n\\\\n"
        f"Sua justificativa DEVE então:\\n"
        f"a) Explicar o porquê do veredito ('Correto' ou 'Incorreto') com base ESTRITAMENTE nas informações contidas nos 'Parágrafos' fornecidos e na sua análise passo-a-passo.\\n"
        f"b) Se possível, referenciar brevemente qual(is) 'Parágrafo(s)' específico(s) (ex: 'Parágrafo 1', 'Parágrafos 2 e 3') sustentam sua conclusão, ou a ausência de informação neles.\\n"
        f"c) NÃO utilize conhecimento externo. Baseie-se APENAS nos textos fornecidos.\\n"
        f"d) Se o veredito for 'Incorreto', mencione sucintamente qual regra (a, b ou c das REGRAS ESTRITAS) levou à sua decisão (ex: 'Incorreto devido à regra b, pois a informação sobre X não foi encontrada no Parágrafo Y').\\n"
        f"Responda em português."
    )

    # User prompt containing only the specific task elements
    user_prompt_content = (
        f"Afirmação: {query}\n\n"
        f"Parágrafos:\n{context_str}"
    )

    # Constructing the Llama 3.1 prompt structure
    # Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3_1/
    # <|begin_of_text|> (special token to signify the start of a prompt)
    # <|start_header_id|>system<|end_header_id|> (system prompt header)
    # System prompt content
    # <|eot_id|> (special token to signify the end of a system prompt)
    # <|start_header_id|>user<|end_header_id|> (user prompt header)
    # User prompt content
    # <|eot_id|> (special token to signify the end of a user prompt)
    # <|start_header_id|>assistant<|end_header_id|> (assistant prompt header, model will generate from here)
    
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt_content}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n" # Model will continue from here
    )
    return prompt

def explain(query: str, list_of_texts: List[str], max_new_tokens: int = 150, temperature: float = 0.5) -> Tuple[str, str]:
    """
    Generates a prediction ("Correto", "Incorreto", or "ERRO_DE_PARSING") and a rationale
    explaining whether the provided texts support the query.
    Uses the meta-llama/Llama-3.1-8B-Instruct model.
    """
    default_prediction = "ERRO_DE_PARSING"
    default_rationale = "Não foi possível gerar a explicação ou parsear o resultado do LLM."

    if not query or not list_of_texts:
        logging.warning("Query and list_of_texts must be provided for explanation.")
        return default_prediction, "Dados de entrada ausentes para o LLM."

    try:
        model, tokenizer = load_explainer_model_and_tokenizer()
        device = model.device if hasattr(model, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

        prompt_text = _format_explanation_prompt(query, list_of_texts)
        logging.debug(f"Generated explanation prompt:\n{prompt_text}")

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=False).to(device)
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature, 
            "top_p": 0.9, # Maintained from previous
            "do_sample": True if temperature > 0 else False, # Sample if temperature is not 0
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id # Important: Llama3 uses specific EOT tokens. Ensure tokenizer.eos_token_id is correctly set for Llama3.
                                              # Often, eos_token_id for Llama3 can be a list of token IDs that signify end of turn.
                                              # Forcing a single one might be okay if it's the primary one.
                                              # The tokenizer for Llama 3 should have this set correctly.
        }
        
        logging.info(f"Generating explanation with {LLM_EXPLAINER_MODEL_ID}...")
        with torch.no_grad():
            # We expect the model to generate "Correto/Incorreto\nJustificativa: ..."
            # We already added "Resultado: " to the prompt.
            output_ids = model.generate(**inputs, **generation_kwargs)
        
        prompt_length = inputs.input_ids.shape[1]
        generated_text_ids = output_ids[0, prompt_length:]
        llm_response_text = tokenizer.decode(generated_text_ids, skip_special_tokens=True).strip()
        
        logging.info(f"LLM Raw Response for explanation: {llm_response_text}")

        # --- PRE-PROCESSING STEP --- 
        # Attempt to remove common prefixes like "Parágrafo X:" or "Parágrafos X-Y:" that the model sometimes adds
        cleaned_llm_response = re.sub(r"^Parágrafo[s]?\s*\d+(?:\s*a\s*\d+|\s*-\s*\d+)?[:.]?\s*", "", llm_response_text, flags=re.IGNORECASE).strip()
        if cleaned_llm_response != llm_response_text:
            logging.info(f"LLM Response after cleaning prefix: {cleaned_llm_response}")
        # --- END PRE-PROCESSING --- 

        # Attempt to parse the structured output FROM THE CLEANED RESPONSE
        # Example: "Correto\nJustificativa: A informação é confirmada pelo Parágrafo 1."
        # Example: "Incorreto\nJustificativa: Nenhum parágrafo confirma a afirmação."

        prediction = default_prediction
        rationale = cleaned_llm_response # Default to cleaned response if parsing fails

        # We added "Resultado: " to the prompt, so model starts generating the veredito.
        # Stricter regex, only accepts "Correto" or "Incorreto" (applied to cleaned_llm_response)
        # Made "Justificativa:" optional and simplified whitespace handling to a single \s*.
        match = re.search(r"^\s*(Correto|Incorreto)\.?\s*(?:Justificativa:\s*)?(.*)", cleaned_llm_response, re.DOTALL | re.IGNORECASE)

        if match:
            extracted_verdict_word = match.group(1).strip().lower()
            extracted_rationale = match.group(2).strip()

            # Map common Portuguese verdicts - now much stricter
            if extracted_verdict_word == "correto":
                prediction = "Correto"
            elif extracted_verdict_word == "incorreto":
                prediction = "Incorreto"
            # No other mappings accepted. If regex matched, it must be one of these.
            # else clause for unmapped (but matched by regex) terms is effectively removed.
            
            # Ensure rationale is present if a valid prediction was made
            if not extracted_rationale:
                 rationale = f"Veredito '{prediction}' parseado, mas explicação não fornecida pelo LLM após a etiqueta 'Justificativa:'."
            else:
                rationale = extracted_rationale
            
            logging.info(f"Parsed LLM Output: Verdict Word='{extracted_verdict_word}', Mapped Prediction='{prediction}', Rationale='{rationale[:100]}...'")
        else:
            logging.warning(f"Could not parse LLM response starting with 'Correto' or 'Incorreto' and optionally 'Justificativa:'. Using cleaned response as rationale. Cleaned response: '{cleaned_llm_response}'")
            # Prediction remains default_prediction ("ERRO_DE_PARSING")
            # Use the cleaned response as the rationale for better error message
            rationale = f"LLM não seguiu o formato esperado ('Correto' ou 'Incorreto'. [Justificativa:] ...). Resposta limpa: {cleaned_llm_response}" 
            # Update the rationale to use the cleaned response for clarity if parsing failed

        # Fallback rationale if everything else failed (e.g., completely empty response)
        if not rationale and prediction == default_prediction: 
            rationale = "A explicação gerada pelo LLM estava vazia ou não foi parseada."
            
        return prediction, rationale

    except Exception as e:
        logging.error(f"Failed to generate explanation for query \'{query[:50]}...\': {e}", exc_info=True)
        return default_prediction, f"Erro ao gerar explicação: {str(e)}"

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
        # Test Case 1
        prediction, explanation = explain(test_query, test_contexts)
        print(f"Test Case 1 - Query: {test_query}")
        print(f"          Contexts: {test_contexts}")
        print(f"  LLM Prediction: {prediction}")
        print(f"  LLM Explanation:\n---\n{explanation}\n---\n")
        
        # Test with empty context
        prediction_empty, explanation_empty_ctx = explain(test_query, [])
        print(f"Test Case Empty Context - Query: {test_query}")
        print(f"  LLM Prediction: {prediction_empty}")
        print(f"  LLM Explanation (Empty Context):\n---\n{explanation_empty_ctx}\n---\n")

        # Test with more complex context
        test_query_2 = "A empresa pública deve seguir o teto remuneratório."
        test_contexts_2 = [
            "Conforme a Constituição, o teto se aplica a todos os entes da administração direta e indireta.",
            "As empresas estatais, contudo, possuem regime jurídico próprio, conforme Lei X.",
            "Decisão do STF no RE YYYY estabeleceu que empresas públicas que não recebem recursos da União para custeio não se submetem ao teto."
        ]
        print(f"\nTest Case 2 - Query: {test_query_2}")
        print(f"          Contexts: {test_contexts_2}\n")
        prediction_2, explanation_2 = explain(test_query_2, test_contexts_2)
        print(f"Test Case 2 - Query: {test_query_2}")
        print(f"          Contexts: {test_contexts_2}")
        print(f"  LLM Prediction: {prediction_2}")
        print(f"  LLM Explanation:\n---\n{explanation_2}\n---\n")

    except RuntimeError as e:
        print(f"RUNTIME ERROR during llm_explainer.py test: {e}")
        print("This often indicates an issue with model loading (e.g., OOM, HuggingFace auth, or network).")
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")

    logging.info("LLM Explainer direct execution test finished.") 