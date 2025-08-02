# test_reply.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load your fine-tuned reply model
model_dir = "reply_checkpoints/"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# 2. Give it a few sample emails
emails = [
    "I need a refund for my recent purchase. Order #12345.",
    "The new dashboard feature you added is amazing!",
    "My login isn’t working and I can’t access my account.",
    "Could you please explain your pricing plans in more detail?"
]

# 3. Generate replies
for email in emails:
    inputs = tokenizer(email, return_tensors="pt", truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=128)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"> Email: {email}\n< Reply: {reply}\n")
