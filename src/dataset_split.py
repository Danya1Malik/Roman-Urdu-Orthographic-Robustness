"""
Stratified Dataset Splitting Module
Project: Robustness of Roman Urdu Sentiment Classification to Orthographic Spelling Variation
"""

import os
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLEAN_REVIEWS_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_reviews.csv"
OUTPUT_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed"
TABLE_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables"


def split_dataset(input_path=CLEAN_REVIEWS_PATH, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    
    logger.info(f"Loading cleaned dataset from {input_path} for stratified split ({train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f})...")
    df = pd.read_csv(input_path, encoding="utf-8")
    
    # Stratified split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df['sentiment']
    )
    
    # Split Temp into Val and Test
    test_rel_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_rel_ratio,
        random_state=random_state,
        stratify=temp_df['sentiment']
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save splits
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    logger.info(f"Saved Train set (N={len(train_df):,}) to {train_path}")
    logger.info(f"Saved Val set   (N={len(val_df):,}) to {val_path}")
    logger.info(f"Saved Test set  (N={len(test_df):,}) to {test_path}")
    
    # Class distribution comparison across splits
    classes = ["very negative", "negative", "neutral", "positive", "very positive"]
    split_summary = []
    for c in classes:
        split_summary.append({
            "Class": c,
            "Train_Count": int((train_df['sentiment'] == c).sum()),
            "Train_Pct": round(float((train_df['sentiment'] == c).mean() * 100), 2),
            "Val_Count": int((val_df['sentiment'] == c).sum()),
            "Val_Pct": round(float((val_df['sentiment'] == c).mean() * 100), 2),
            "Test_Count": int((test_df['sentiment'] == c).sum()),
            "Test_Pct": round(float((test_df['sentiment'] == c).mean() * 100), 2),
        })
        
    summary_df = pd.DataFrame(split_summary)
    summary_df.to_csv(os.path.join(TABLE_DIR, "dataset_splits_distribution.csv"), index=False)
    
    meta = {
        "random_state": random_state,
        "total_records": len(df),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "class_distributions": split_summary
    }
    
    with open(os.path.join(TABLE_DIR, "dataset_splits_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    return train_df, val_df, test_df, meta


if __name__ == "__main__":
    split_dataset()
