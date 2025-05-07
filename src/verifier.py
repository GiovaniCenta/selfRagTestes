import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import Tuple, Optional, Dict, Any
import numpy as np # For softmax

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - VERIFIER - %(levelname)s - %(message)s')

# --- Constants --- 
BASE_MODEL_ID = "rufimelo/Legal-BERTimbau-sts-large"
LORA_ADAPTER_PATH = "./legalbert_lora" # Path to the LoRA adapter directory
NUM_LABELS = 2 # CONFERE / NÃO CONFERE
ID2LABEL = {0: "NÃO CONFERE", 1: "CONFERE"}
LABEL2ID = {"NÃO CONFERE": 0, "CONFERE": 1}

# --- Cached Model and Tokenizer --- 
_verifier_model_instance: Optional[Any] = None # Can be base or PeftModel
_verifier_tokenizer_instance: Optional[AutoTokenizer] = None

def load_verifier_model_and_tokenizer(
    base_model_id: str = BASE_MODEL_ID,
    lora_adapter_path: Optional[str] = LORA_ADAPTER_PATH,
    num_labels: int = NUM_LABELS
) -> Tuple[Any, AutoTokenizer]:
    """
    Loads the base sequence classification model and tokenizer.
    If a LoRA adapter path is provided and exists, applies the LoRA adapter.
    Caches the model and tokenizer.
    """
    global _verifier_model_instance, _verifier_tokenizer_instance

    if _verifier_model_instance is not None and _verifier_tokenizer_instance is not None:
        logging.debug("Returning cached verifier model and tokenizer.")
        return _verifier_model_instance, _verifier_tokenizer_instance

    logging.info(f"Loading verifier tokenizer for: {base_model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Or a specific pad token if model prefers
            logging.info(f"Set tokenizer pad_token to: {tokenizer.pad_token}")
        _verifier_tokenizer_instance = tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{base_model_id}': {e}", exc_info=True)
        raise RuntimeError(f"Tokenizer load failed: {e}") from e

    logging.info(f"Loading base verifier model: {base_model_id} for sequence classification ({num_labels} labels).")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        # Quantization settings if desired (may need adjustment for sequence classification)
        # For Legal-BERTimbau, 4-bit might be aggressive for classification. 
        # Consider full precision or 8-bit if performance issues arise.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, # Or False for no quantization
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if torch.cuda.is_available() else None # Only quantize if CUDA is available

        model_kwargs = {
            "num_labels": num_labels,
            "id2label": ID2LABEL,
            "label2id": LABEL2ID,
            "torch_dtype": torch.bfloat16 if device == 'cuda' else torch.float32,
            "low_cpu_mem_usage": True
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto" # Let accelerate handle device mapping for quantized models
        else:
            model_kwargs["device_map"] = None # Will be moved to device later if not quantized

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            **model_kwargs
        )

        if not quantization_config and device == 'cuda':
            model = model.to(device)

        logging.info(f"Base verifier model loaded on device: {model.device if hasattr(model, 'device') else device}")

        # Apply LoRA adapter if path exists
        if lora_adapter_path and os.path.exists(lora_adapter_path) and os.path.isdir(lora_adapter_path):
            logging.info(f"Found LoRA adapter at: {lora_adapter_path}. Attempting to load.")
            try:
                # Check if it's a PEFT model already saved with save_pretrained
                if any(fname.startswith('adapter_config.json') for fname in os.listdir(lora_adapter_path)):
                    model = PeftModel.from_pretrained(model, lora_adapter_path)
                    logging.info(f"Successfully applied LoRA adapter from {lora_adapter_path}.")
                    if hasattr(model, 'merge_and_unload') and False: # Optional: merge if not training further
                        logging.info("Merging LoRA weights into base model.")
                        model = model.merge_and_unload()
                else:
                    logging.warning(f"LoRA adapter path '{lora_adapter_path}' does not seem to contain a valid PEFT adapter (missing adapter_config.json).")        
            except Exception as peft_err:
                logging.error(f"Failed to load or apply LoRA adapter from '{lora_adapter_path}': {peft_err}", exc_info=True)
                # Continue with base model if LoRA fails
        else:
            logging.info("No LoRA adapter path provided or path does not exist. Using base model.")
        
        _verifier_model_instance = model
        _verifier_model_instance.eval() # Set to evaluation mode

    except Exception as e:
        logging.error(f"Failed to load verifier model '{base_model_id}': {e}", exc_info=True)
        raise RuntimeError(f"Verifier model load failed: {e}") from e

    return _verifier_model_instance, _verifier_tokenizer_instance

def classify(query: str, text: str) -> Tuple[str, float]:
    """
    Classifies if the text supports the query (CONFERE / NÃO CONFERE)
    using the loaded Legal-BERTimbau model (potentially with LoRA).

    Args:
        query: The query or claim string.
        text: The evidence text (e.g., top-1 reranked document chunk).

    Returns:
        A tuple containing the predicted label (str) and its confidence (float).
        Returns ("ERRO", 0.0) if classification fails.
    """
    if not query or not text:
        logging.warning("Query and text must be provided for classification.")
        return "ERRO", 0.0

    try:
        model, tokenizer = load_verifier_model_and_tokenizer()
        device = model.device if hasattr(model, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

        logging.debug(f"Classifying query: '{query[:50]}...' with text: '{text[:100]}...'")
        
        # Prepare input for sequence classification (query, text pair)
        # Explicitly set max_length to 512 to avoid potential OverflowError
        inputs = tokenizer(
            query, 
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512 # Explicitly set a safe max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Apply softmax to get probabilities
        # Convert to float32 before CPU/NumPy conversion as NumPy doesn't support bfloat16
        probabilities = torch.softmax(logits, dim=-1).to(torch.float32).cpu().numpy()[0]
        
        predicted_class_id = np.argmax(probabilities)
        predicted_label = model.config.id2label[predicted_class_id]
        confidence = float(probabilities[predicted_class_id])
        
        logging.info(f"Classification result: Label='{predicted_label}', Confidence={confidence:.4f}")
        return predicted_label, confidence

    except Exception as e:
        logging.error(f"Failed to classify query '{query[:50]}...': {e}", exc_info=True)
        return "ERRO", 0.0

# Example (for testing if run directly)
if __name__ == '__main__':
    # Ensure ./legalbert_lora directory exists if you want to test LoRA loading
    # You might need to create a dummy adapter_config.json and adapter_model.bin for this test
    # e.g., by saving a dummy PeftModel if you don't have the real one.
    
    # Test Case 1: Expected CONFERE
    test_query_1 = "O prazo para recurso é de 15 dias."
    test_text_1 = "Conforme o artigo 10, o prazo recursal será de quinze dias úteis."
    print(f"Test Case 1 - Query: {test_query_1}")
    print(f"            Text: {test_text_1}")
    label1, conf1 = classify(test_query_1, test_text_1)
    print(f"--> Result: {label1}, Confidence: {conf1:.4f}\n")

    # Test Case 2: Expected NÃO CONFERE (Contradiction or Irrelevant)
    test_query_2 = "O contrato foi assinado em 2023."
    test_text_2 = "O instrumento contratual foi celebrado em janeiro de 2022."
    print(f"Test Case 2 - Query: {test_query_2}")
    print(f"            Text: {test_text_2}")
    label2, conf2 = classify(test_query_2, test_text_2)
    print(f"--> Result: {label2}, Confidence: {conf2:.4f}\n")

    # Test Case 3: Expected NÃO CONFERE (Insufficient)
    test_query_3 = "Houve dano moral."
    test_text_3 = "A petição inicial alega a ocorrência de diversos transtornos."
    print(f"Test Case 3 - Query: {test_query_3}")
    print(f"            Text: {test_text_3}")
    label3, conf3 = classify(test_query_3, test_text_3)
    print(f"--> Result: {label3}, Confidence: {conf3:.4f}\n")

    # Test with a potentially longer text to check truncation
    long_text = "Este é um texto muito longo que certamente excederá o limite máximo de tokens do modelo " \
                "BERT, que geralmente é de 512 tokens. Vamos repetir esta frase várias vezes para garantir " \
                "que ela seja longa o suficiente. Repetindo: Este é um texto muito longo que certamente " \
                "excederá o limite máximo de tokens. Mais uma vez para ter certeza absoluta que o texto é extenso. " * 5
    test_query_4 = "O texto é longo."
    print(f"Test Case 4 (Long Text) - Query: {test_query_4}")
    # print(f"            Text: {long_text[:100]}...")
    label4, conf4 = classify(test_query_4, long_text)
    print(f"--> Result: {label4}, Confidence: {conf4:.4f}\n")

    logging.info("Verifier direct execution test finished.") 