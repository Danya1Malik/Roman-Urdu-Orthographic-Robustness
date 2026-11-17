"""
External Dataset Validation Module (RUECD on Daraz.pk)
Evaluates generalizability of models on an independent external Roman Urdu e-commerce dataset.
Handles 5-class to 3-class collapsing and reports dataset overlap.
"""

import os
import json
import logging
import pandas as pd
import numpy as np

from src.evaluate import compute_classification_metrics
from src.perturbation_engine import PerturbationEngine

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RUECD_RAW_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\Dataset\RUECD.csv"
RUECD_CLEAN_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_ruecd.csv"
RUWV_CLEAN_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_reviews.csv"
TABLE_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables"


def prepare_ruecd_dataset():
    logger.info("Loading and standardizing RUECD dataset...")
    df = pd.read_csv(RUECD_RAW_PATH)
    df = df.rename(columns={"Reviews": "review", "Sentiment": "sentiment_raw"})
    
    # Label mapping: 0 -> negative, 1 -> positive, 2 -> neutral
    label_map = {0: "negative", 1: "positive", 2: "neutral"}
    df["sentiment"] = df["sentiment_raw"].map(label_map)
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"] != ""].copy()
    
    # Calculate textual overlap with primary dataset RUWV-NSR
    df_ruwv = pd.read_csv(RUWV_CLEAN_PATH)
    ruwv_texts = set(df_ruwv["review"].astype(str).str.strip().str.lower())
    df["is_overlap"] = df["review"].str.lower().isin(ruwv_texts)
    
    overlap_count = int(df["is_overlap"].sum())
    overlap_pct = (overlap_count / len(df)) * 100
    
    logger.info(f"RUECD total reviews: {len(df):,}")
    logger.info(f"RUECD class distribution:\n{df['sentiment'].value_counts().to_dict()}")
    logger.info(f"Overlap with RUWV-NSR: {overlap_count} reviews ({overlap_pct:.2f}%)")
    
    os.makedirs(os.path.dirname(RUECD_CLEAN_PATH), exist_ok=True)
    df.to_csv(RUECD_CLEAN_PATH, index=False, encoding="utf-8")
    
    overlap_report = {
        "ruecd_total": len(df),
        "ruecd_sentiment_counts": df["sentiment"].value_counts().to_dict(),
        "overlap_count": overlap_count,
        "overlap_pct": float(overlap_pct)
    }
    with open(os.path.join(TABLE_DIR, "ruecd_overlap_report.json"), "w", encoding="utf-8") as f:
        json.dump(overlap_report, f, indent=2)
        
    return df, overlap_report


def evaluate_model_on_ruecd(predict_fn, model_name="model", disjoint_only=True):
    """
    Evaluates predict_fn (which takes list of review strings and outputs 5-class or 3-class predictions)
    on RUECD across Clean and Perturbation conditions.
    """
    df = pd.read_csv(RUECD_CLEAN_PATH)
    if disjoint_only:
        df = df[~df["is_overlap"]].copy().reset_index(drop=True)
        
    # Sample 3,000 reviews for rapid standardized evaluation
    df_sample = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)
    
    engine = PerturbationEngine()
    conditions = ["clean", "mild", "moderate", "severe"]
    results = []
    
    # 5-class to 3-class collapse mapping
    collapse_5to3 = {
        "very positive": "positive",
        "positive": "positive",
        "neutral": "neutral",
        "negative": "negative",
        "very negative": "negative"
    }
    
    for cond in conditions:
        if cond == "clean":
            perturbed_texts = [str(t) for t in df_sample["review"]]
        elif cond == "mild":
            perturbed_texts = [engine.perturb_text(str(t), rate=0.20, distance_strategy="low", seed=42+i)[0] for i, t in enumerate(df_sample["review"])]
        elif cond == "moderate":
            perturbed_texts = [engine.perturb_text(str(t), rate=0.50, distance_strategy="random", seed=42+i)[0] for i, t in enumerate(df_sample["review"])]
        elif cond == "severe":
            perturbed_texts = [engine.perturb_text(str(t), rate=0.85, distance_strategy="high", seed=42+i)[0] for i, t in enumerate(df_sample["review"])]
            
        raw_preds = predict_fn(perturbed_texts)
        # Collapse if 5-class
        preds_3 = [collapse_5to3.get(p, p) for p in raw_preds]
        
        metrics = compute_classification_metrics(df_sample["sentiment"], preds_3, labels=["negative", "neutral", "positive"])
        results.append({
            "model": model_name,
            "dataset": "RUECD (Daraz.pk)",
            "condition": cond,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "n_samples": len(df_sample)
        })
        
    res_df = pd.DataFrame(results)
    return res_df


if __name__ == "__main__":
    prepare_ruecd_dataset()
