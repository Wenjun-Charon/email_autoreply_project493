#!/usr/bin/env python3
import os
import sys
import click

# Ensure project root is on Python path for src imports\ nPROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.classify.train import train_classifier
from src.reply.train import train_reply_model

@click.group()
def cli():
    """Email Autoreply Service CLI"""
    pass

@cli.command()
@click.option('--data-dir', default='data/', help='Directory containing train.csv & valid.csv for classification')
@click.option('--model-name', default='distilbert-base-uncased', help='Pretrained Hugging Face classification model')
@click.option('--output-dir', default='checkpoints/', help='Directory to save the classification model')
def train(data_dir, model_name, output_dir):
    """Train the email classification model"""
    train_classifier(data_dir, model_name, output_dir)

@cli.command(name='train_reply')
@click.option('--data-dir', default='data/', help='Directory containing train_reply.csv & valid_reply.csv for replies')
@click.option('--model-name', default='t5-base', help='Pretrained Hugging Face seq2seq model for reply generation')
@click.option('--output-dir', default='reply_checkpoints/', help='Directory to save the reply generation model')
def train_reply(data_dir, model-name, output_dir):
    """Train the reply generation model on email→reply pairs"""
    train_reply_model(data_dir, model_name, output_dir)

if __name__ == '__main__':
    cli()
