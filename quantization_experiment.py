# quantization_experiment.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.quantization import quantize_dynamic

def load_and_quantize_model(model_name: str = 'gpt2'):
    """
    Loads a pre-trained model and applies dynamic quantization.

    Parameters:
    - model_name: str, the model identifier from Hugging Face's model repository.

    Returns:
    - quantized_model: The quantized version of the pre-trained model.
    """
    # Load the pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Apply dynamic quantization to the model
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return quantized_model

if __name__ == "__main__":
    model_name = 'gpt2'  # You can replace this with other model names as needed
    quantized_model = load_and_quantize_model(model_name)
    print(f"Quantized model loaded successfully with reduced size: {model_name}")
