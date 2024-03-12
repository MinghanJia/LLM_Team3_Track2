# distillation_experiment.py

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from datasets import load_dataset

def train_student_model(teacher_model, student_model, tokenizer, dataset, device='cpu'):
    """
    Trains a student model using knowledge distillation from a teacher model.

    Parameters:
    - teacher_model: The pre-trained larger model acting as the teacher.
    - student_model: The smaller model to be trained.
    - tokenizer: Tokenizer for encoding the text data.
    - dataset: Training dataset.
    - device: The device to run the training on ('cpu' or 'cuda').
    """

    teacher_model.to(device).eval()
    student_model.to(device).train()

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(student_model.parameters(), lr=5e-5)

    for epoch in range(3):  # Number of epochs can be adjusted
        for batch in dataloader:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs, labels=inputs['input_ids']).logits
            
            student_outputs = student_model(**inputs, labels=inputs['input_ids']).logits
            
            loss = torch.nn.functional.kl_div(
                input=torch.nn.functional.log_softmax(student_outputs, dim=-1),
                target=torch.nn.functional.softmax(teacher_outputs, dim=-1),
                reduction='batchmean'
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

if __name__ == "__main__":
    # Load the pre-trained teacher and student models
    teacher_model = GPT2LMHeadModel.from_pretrained('gpt3')  # Placeholder, GPT-3 loading should be adjusted
    student_model = GPT2LMHeadModel.from_pretrained('gpt2')  # Example smaller model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load a dataset
    dataset = load_dataset("text", data_files="path/to/your/dataset.txt")['train']

    # Train the student model
    train_student_model(teacher_model, student_model, tokenizer, dataset)
