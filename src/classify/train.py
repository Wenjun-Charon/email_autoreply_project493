#!/usr/bin/env python3

import os
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

def train_classifier(data_dir: str, model_name: str, output_dir: str):
    """
    Fine-tune a sequence classification model for email categorization.

    Args:
        data_dir: Directory containing 'train.csv' and 'valid.csv'.
        model_name: Hugging Face identifier for the base model (e.g., 'distilbert-base-uncased').
        output_dir: Directory to save the fine-tuned model and tokenizer.

    Workflow:
        1. Load and encode the datasets.
        2. Tokenize emails with fixed-length padding.
        3. Configure training arguments.
        4. Initialize Trainer, train, and evaluate.
        5. Save the trained model and tokenizer.
    """
    # 1. Load datasets from CSV files
    files = {
        "train": os.path.join(data_dir, "train.csv"),
        "valid": os.path.join(data_dir, "valid.csv"),
    }
    dataset = load_dataset("csv", data_files=files)

    # 2. Build label mapping and encode string labels to IDs
    label_list = dataset["train"].unique("label")
    label2id = {label: idx for idx, label in enumerate(label_list)}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    dataset = dataset.map(encode_labels)

    # 3. Initialize tokenizer and tokenize emails
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 128

    def tokenize_batch(batch):
        return tokenizer(
            batch["email"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["email"]
    )

    # 4. Set dataset format for PyTorch
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # 5. Initialize sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list)
    )

    # 6. Configure training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
    )

    # 7. Load accuracy metric and define compute_metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"accuracy": metric.compute(predictions=preds, references=labels)["accuracy"]}

    # 8. Initialize Trainer and train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        compute_metrics=compute_metrics
    )
    trainer.train()

    # 9. Save the fine-tuned model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
