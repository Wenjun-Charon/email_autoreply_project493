#!/usr/bin/env python3
"""
split_reply_dataset.py

Pair each incoming email with its corresponding reply based on the 'subject' column,
then split into training and validation sets for reply-generation fine-tuning.
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 1. Input file (must be in project root)
    input_file = "dataset.csv"
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Could not find '{input_file}' in the project root.")
    
    # 2. Load the full dataset
    df = pd.read_csv(input_file)  # expects columns 'subject' and 'message_body'
    
    # 3. Normalize subjects: strip leading "Re: " from reply subjects
    df["base_subject"] = df["subject"].str.replace(r"^Re:\s*", "", regex=True)
    
    # 4. Group by base_subject and extract one email/reply pair per group
    pairs = []
    for subj, group in df.groupby("base_subject"):
        originals = group[group["subject"] == subj]
        replies  = group[group["subject"].str.match(rf"^Re:\s*{re.escape(subj)}")]
        if not originals.empty and not replies.empty:
            pairs.append({
                "email": originals.iloc[0]["message_body"],
                "reply": replies.iloc[0]["message_body"]
            })
    
    pairs_df = pd.DataFrame(pairs)
    print(f"Total paired examples: {len(pairs_df)}")
    
    # 5. Split into train (80%) and valid (20%)
    train_df, valid_df = train_test_split(
        pairs_df,
        test_size=0.2,
        random_state=42
    )
    
    # 6. Write out to data/
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_reply.csv", index=False)
    valid_df.to_csv("data/valid_reply.csv", index=False)
    print(f"Saved {len(train_df)} train pairs to data/train_reply.csv")
    print(f"Saved {len(valid_df)} valid pairs to data/valid_reply.csv")

if __name__ == "__main__":
    main()
