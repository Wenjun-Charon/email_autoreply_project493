#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmailClassifier:

    def __init__(self, model_path: str, num_labels: int):
        """
        Initialize the classifier by loading model and tokenizer from a directory.

        Args:
            model_path: Path to directory with pretrained model and tokenizer.
            num_labels: Number of label classes for the classification head.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )

    @classmethod
    def load(cls, model_path: str, num_labels: int):
        """
        Load a trained classifier from disk.

        Args:
            model_path: Directory where model and tokenizer are saved.
            num_labels: Number of label classes used during training.

        Returns:
            An instance of EmailClassifier.
        """
        return cls(model_path, num_labels)

    def save(self, save_directory: str):
        """
        Save the trained model and tokenizer to a directory.

        Args:
            save_directory: Path to save the model and tokenizer.
        """
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def predict(self, texts: list[str]):
        """
        Predict labels for a list of email texts.

        Args:
            texts: A list of email body strings to classify.

        Returns:
            A tuple (labels, confidences) where:
              - labels: List of predicted label IDs.
              - confidences: List of max softmax probabilities per text.
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        # Forward pass
        outputs = self.model(**inputs)
        # Compute probabilities
        probs = torch.softmax(outputs.logits, dim=-1)
        # Get predicted label IDs and confidences
        label_ids = torch.argmax(probs, dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()
        return label_ids, confidences
