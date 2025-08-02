# Email Autoreply Service

An end-to-end system for automating customer support emails. It consists of two core components:

1. **Email Classification** – fine-tune a DistilBERT model to categorize incoming emails into business types (e.g., inquiry, issue, suggestion).
2. **Reply Generation** – fine-tune a T5‐style Seq2Seq model to generate appropriate replies based on the email content.

## Project Structure

```
email_autoreply_project/
├── cli.py                     # Main CLI for training and serving
├── README.md                  # This documentation file
├── requirements.txt           # Python dependencies
├── dataset.csv                # Raw data (email threads)
├── split_dataset.py           # Prepare train/valid CSVs for classification
├── split_reply_dataset.py     # Prepare train_reply/valid_reply CSVs for generation
├── test_classify.py           # Smoke-test classification model
├── test_reply.py              # Smoke-test reply model
│
├── data/                      # Generated CSVs
│   ├── train.csv              # Classification training data
│   ├── valid.csv              # Classification validation data
│   ├── train_reply.csv        # Reply-generation training data
│   └── valid_reply.csv        # Reply-generation validation data
│
├── checkpoints/               # Saved classification model
│
├── reply_checkpoints/         # Saved reply-generation model
│
└── src/                       # Source code
    ├── classify/
    │   ├── __init__.py
    │   ├── train.py           # Classification training script
    │   └── model.py           # EmailClassifier wrapper
    │
    ├── reply/
    │   ├── __init__.py
    │   ├── train.py           # Reply-generation training script
    │   └── generator.py       # Inference wrapper for reply model
    │
    └── api/
        ├── __init__.py
        └── app.py             # FastAPI service definition
```

## Prerequisites

- Python 3.8+ (tested on 3.11)
- Git
- (Optional) CUDA-enabled GPU for faster training

## Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url> email_autoreply_project
   cd email_autoreply_project
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   # .\venv\Scripts\activate  # Windows
   ```

3. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Data Preparation

1. **Place your raw **`` in the project root. It must contain at least:

   - `subject` (string)
   - `message_body` (string)
   - `email_types` (e.g. `['inquiry']`)

2. **Generate classification splits**:

   ```bash
   python split_dataset.py
   ```

   - Outputs `data/train.csv` and `data/valid.csv` for classification.

3. **Generate reply-generation splits**:

   ```bash
   python split_reply_dataset.py
   ```

   - Outputs `data/train_reply.csv` and `data/valid_reply.csv` for reply model.

## Training

### 1. Email Classification (optional)

```bash
python cli.py train \
  --data-dir data/ \
  --model-name distilbert-base-uncased \
  --output-dir checkpoints/
```

- Trains a DistilBERT classifier on `data/train.csv`/`valid.csv`.
- Saves the best model to `checkpoints/`.

### 2. Reply Generation

```bash
python cli.py train-reply \
  --data-dir data/ \
  --model-name t5-base \
  --output-dir reply_checkpoints/
```

- Trains a T5-style Seq2Seq model on `data/train_reply.csv`/`valid_reply.csv`.
- Uses a `generate reply:` task prefix for improved learning.
- Saves the fine-tuned model to `reply_checkpoints/`.

## Testing Models

- **Classification**:
  ```bash
  python test_classify.py
  ```
- **Reply Generation**:
  ```bash
  python test_reply.py
  ```

Inspect printed outputs to verify correctness.

## Serving via FastAPI

1. **Launch the API**

   ```bash
   python cli.py serve --host 127.0.0.1 --port 8000
   ```

2. **Test endpoints**

   - **Classify**:
     ```bash
     curl -X POST http://127.0.0.1:8000/classify \
       -H "Content-Type: application/json" \
       -d '{"email":"My account is locked."}'
     ```
   - **Reply**:
     ```bash
     curl -X POST http://127.0.0.1:8000/reply \
       -H "Content-Type: application/json" \
       -d '{"email":"I need help with my order."}'
     ```

## Notes & Troubleshooting

- **No **``** needed**: All inference is local. You can delete any `.env` related code.
- **Click commands**: functions with underscores become hyphens on CLI (`train-reply`).
- **Transformers & Legacy Args**: Scripts use legacy training arguments for compatibility with older `transformers` versions.

## License

MIT License — feel free to adapt and extend for your own use.

