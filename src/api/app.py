#!/usr/bin/env python3

from fastapi import FastAPI
from pydantic import BaseModel
from src.classify.model import EmailClassifier
from src.reply.generator import generate_reply

NUM_LABELS = 4

def create_app():
    """
    Create and configure the FastAPI application with classification and reply endpoints.
    """
    app = FastAPI(title="Email Autoreply Service")

    # Load the classification model
    classifier = EmailClassifier.load("checkpoints/", num_labels=NUM_LABELS)

    # Request body schema for both endpoints
    class EmailRequest(BaseModel):
        email: str

    @app.post("/classify")
    def classify(req: EmailRequest):
        """
        Classify an incoming email into one of the predefined categories.
        """
        labels, scores = classifier.predict([req.email])
        return {"label_id": labels[0], "confidence": scores[0]}

    @app.post("/reply")
    def reply(req: EmailRequest):
        """
        Generate an automatic reply for the incoming email.
        """
        reply_text = generate_reply(req.email)
        return {"reply": reply_text}

    return app

