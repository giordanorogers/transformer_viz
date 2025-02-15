"""
The model handling logic.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(
    model_name,
    output_hidden_states=True,
    output_attentions=True,
    return_dict_in_generate=True
)
model.eval()

def get_activations(prompt):
    """
    Runs a prompt through GPT-2 and returns:
        - Tokenized inputs
        - Logits
        - Hidden states (activations)
        - Attentions
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    logits = outputs.logits  # shape: (batch, sequence_length, vocab_size)
    hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors
    attentions = outputs.attentions  # tuple of attention matrices per layer
    return inputs, logits, hidden_states, attentions
