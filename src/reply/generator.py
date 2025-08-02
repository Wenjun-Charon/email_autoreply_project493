#!/usr/bin/env python3

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

dir_path = os.getenv("REPLY_MODEL_PATH", "reply_checkpoints")

# Load tokenizer and model once at import time
tokenizer = AutoTokenizer.from_pretrained(dir_path)
model     = AutoModelForSeq2SeqLM.from_pretrained(dir_path)


def generate_reply(email: str, label: str = None) -> str:
    """
    Generate a customer service reply for a given email text.

    Args:
        email: The incoming customer email body.
        label: Optional classification label (unused in this generator).

    Returns:
        A generated reply string.
    """
    # Tokenize the input email
    inputs = tokenizer(
        email,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    # Generate reply with beam search and no repeated n-grams
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    # Decode and return the generated text
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply
