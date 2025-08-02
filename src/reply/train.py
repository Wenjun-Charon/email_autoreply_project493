import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


def train_reply_model(data_dir: str, model_name: str, output_dir: str):
    """
    Fine-tune a sequence-to-sequence model to generate customer service replies.

    Args:
        data_dir: Directory containing 'train_reply.csv' and 'valid_reply.csv'.
        model_name: Hugging Face identifier for the base Seq2Seq model (e.g., 't5-base').
        output_dir: Directory to save the fine-tuned model and tokenizer.

    Workflow:
        1. Load paired email/reply CSVs for training and validation.
        2. Prepend a task prefix to each email and tokenize inputs and labels.
        3. Configure training arguments for multiple epochs and legacy support.
        4. Initialize Seq2SeqTrainer and run training.
        5. Save the trained model and tokenizer.
    """
    # 1. Load datasets
    files = {
        "train": os.path.join(data_dir, "train_reply.csv"),
        "valid": os.path.join(data_dir, "valid_reply.csv"),
    }
    dataset = load_dataset("csv", data_files=files)

    # 2. Initialize tokenizer and define max length
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 128

    # 3. Preprocessing: add task prefix and tokenize inputs and targets
    def preprocess(batch):
        # Add a clear instruction prefix
        prefixed = ["generate reply: " + text for text in batch["email"]]
        inputs = tokenizer(
            prefixed,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        # Tokenize the target replies
        targets = tokenizer(
            batch["reply"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        # Set labels for computing Seq2Seq loss
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["email", "reply"]
    )

    # 4. Format dataset for PyTorch
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # 5. Load base Seq2Seq model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 6. Define training arguments (legacy style for compatibility)
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=5,                # Increased epochs for better learning
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
    )

    # 7. Initialize Trainer and train
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        tokenizer=tokenizer
    )
    trainer.train()

    # 8. Save the fine-tuned model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
