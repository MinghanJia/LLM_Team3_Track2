# pruning_experiment.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.utils import prune

def load_and_prune_model(model_name: str = 'gpt2', pruning_amount: float = 0.2):
    """
    Loads a pre-trained model and applies pruning to reduce the number of parameters.

    Parameters:
    - model_name: str, the identifier of the model to load.
    - pruning_amount: float, the fraction of connections to prune (remove) from each layer.

    Returns:
    - model: The pruned version of the pre-trained model.
    """
    # Load the pre-trained model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Apply pruning to each linear layer in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            # To make the pruning permanent, remove the re-parametrization
            prune.remove(module, 'weight')
    
    return model

if __name__ == "__main__":
    model_name = 'gpt2'  # Example model, you can replace it with other models as needed
    pruning_amount = 0.2  # Example pruning 20% of the connections
    
    pruned_model = load_and_prune_model(model_name, pruning_amount)
    print(f"Model '{model_name}' pruned successfully. Pruning amount: {pruning_amount * 100}%")
