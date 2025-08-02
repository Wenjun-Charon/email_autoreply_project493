#!/usr/bin/env python3
"""
split_dataset.py

Prepare train.csv and valid.csv for classification by:
  - Loading the original dataset.csv
  - Removing reply entries (subjects starting with "Re:")
  - Cleaning and mapping email_types into a single-label column
  - Splitting 80/20 stratified by label
"""
import os
import ast
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    input_file = "dataset.csv"
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Could not find '{input_file}' in the project root.")

    # 1. Load the full dataset
    df = pd.read_csv(input_file)

    # 2. Remove replies: drop any row whose subject starts with "Re:"
    mask = ~df["subject"].str.match(r"^Re:\s*")
    df = df[mask].copy()

    # 3. Rename the message body column for classification
    df = df.rename(columns={"message_body": "email"})

    # 4. Clean email_types into a single-label string
    def clean_types(x):
        try:
            # Parse the list literal, join multiple types with '|'
            types = ast.literal_eval(x)
            if isinstance(types, list):
                return "|".join(str(t) for t in types)
        except Exception:
            pass
        # Fallback: strip brackets/quotes
        return x.strip("[]\"' ")

    df["label"] = df["email_types"].apply(clean_types)

    # 5. Select only the columns needed for classification
    df = df[["email", "label"]]

    # 6. Split into train (80%) and valid (20%) with stratification
    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    # 7. Save to data/ directory
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    valid_df.to_csv("data/valid.csv", index=False)

    print(f"Saved {len(train_df)} examples to data/train.csv")
    print(f"Saved {len(valid_df)} examples to data/valid.csv")


if __name__ == "__main__":
    main()
