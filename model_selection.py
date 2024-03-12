# model_selection.py

from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_pretrained_llm(model_name: str = 'gpt2'):
    """
    Loads a pre-trained Large Language Model (LLM) and its tokenizer.

    Parameters:
    - model_name: str, the identifier of the model to load. Defaults to 'gpt2'.

    Returns:
    - model: The pre-trained GPT-2 model.
    - tokenizer: The tokenizer for the GPT-2 model.
    """
    # Load the tokenizer for the specified model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    print(f"Model and tokenizer for '{model_name}' loaded successfully.")
    
    return model, tokenizer

if __name__ == "__main__":
    # Example model name: 'gpt2'. You can also use 'gpt2-medium', 'gpt2-large', or 'gpt2-xl' for larger models.
    model_name = 'gpt2'
    
    # Load the model and tokenizer
    model, tokenizer = load_pretrained_llm(model_name)
