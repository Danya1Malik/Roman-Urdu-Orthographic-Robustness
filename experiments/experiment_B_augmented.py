"""
Experiment B: Data Augmentation & Robust Training
Investigates whether augmenting training data with spelling perturbations
improves test-time robustness under orthographic variation.
Compares Clean-Trained vs Augmented-Trained models across Clean, Mild, Moderate, and Severe conditions.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perturbation_engine import PerturbationEngine
from src.train import (
    train_tfidf,
    train_transformer,
    CHECKPOINT_DIR
)
from src.evaluate import (
    compute_classification_metrics,
    CLASS_NAMES_5
)
from experiments.experiment_A_zeroshot import predict_transformer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed"
RESULTS_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


def generate_augmented_training_data(train_df, augmentation_ratio=0.50, seed=42):
    """
    Creates an augmented training set by creating perturbed copies of a subset of training reviews.
    """
    engine = PerturbationEngine()
    logger.info(f"Generating augmented training data from {len(train_df):,} base training records...")
    
    n_to_augment = int(len(train_df) * augmentation_ratio)
    sample_to_augment = train_df.sample(n=n_to_augment, random_state=seed).copy()
    
    augmented_rows = []
    for idx, row in sample_to_augment.iterrows():
        # Apply moderate variation (rate=0.40, random strategy)
        p_text, _ = engine.perturb_text(str(row["review"]), rate=0.40, distance_strategy="random", seed=seed + idx)
        new_row = row.copy()
        new_row["review"] = p_text
        augmented_rows.append(new_row)
        
    augmented_df = pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)
    augmented_df = augmented_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    aug_path = os.path.join(DATA_DIR, "train_augmented.csv")
    augmented_df.to_csv(aug_path, index=False, encoding="utf-8")
    logger.info(f"Saved augmented train set (N={len(augmented_df):,}) to {aug_path}")
    return augmented_df


def run_experiment_b():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    
    train_aug_path = os.path.join(DATA_DIR, "train_augmented.csv")
    if not os.path.exists(train_aug_path):
        train_aug_df = generate_augmented_training_data(train_df)
    else:
        train_aug_df = pd.read_csv(train_aug_path)
        
    models_aug = {}
    
    # 1. TF-IDF Augmented
    logger.info("Training Augmented TF-IDF Baseline...")
    tfidf_aug, _ = train_tfidf(train_aug_df, val_df, save_name="tfidf_augmented.joblib")
    models_aug["TF-IDF (Augmented)"] = {
        "predict_fn": lambda texts: tfidf_aug.predict([str(t) for t in texts])
    }
    
    # 2. XLM-RoBERTa LoRA Augmented
    logger.info("Training Augmented XLM-RoBERTa LoRA...")
    lora_aug, lora_tok, _ = train_transformer(
        model_name="xlm-roberta-base",
        use_lora=True,
        train_df=train_aug_df,
        val_df=val_df,
        epochs=3,
        batch_size=32,
        lr=2e-4,
        max_length=64,
        save_name="lora_xlm_roberta_augmented",
        device=device
    )
    models_aug["XLM-R LoRA (Augmented)"] = {
        "predict_fn": lambda texts: predict_transformer(lora_aug, lora_tok, texts, device)
    }
    
    # Evaluate across canonical conditions
    conditions = ["clean", "mild", "moderate", "severe"]
    results = []
    
    for cond in conditions:
        test_df = pd.read_csv(os.path.join(DATA_DIR, f"test_{cond}.csv"))
        y_true = test_df["sentiment"].tolist()
        
        for name, m_info in models_aug.items():
            y_pred = m_info["predict_fn"](test_df["review"].tolist())
            metrics = compute_classification_metrics(y_true, y_pred)
            results.append({
                "model": name,
                "training_regime": "Augmented",
                "condition": cond,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"]
            })
            
    res_df = pd.DataFrame(results)
    
    # Load clean-trained results from Experiment A to compare
    exp_a_path = os.path.join(TABLE_DIR, "experiment_A_canonical_results.csv")
    if os.path.exists(exp_a_path):
        exp_a_df = pd.read_csv(exp_a_path)
        exp_a_df["training_regime"] = "Clean-Only"
        combined = pd.concat([
            exp_a_df[exp_a_df["model"].isin(["TF-IDF + LR", "XLM-RoBERTa (LoRA)"])],
            res_df
        ], ignore_index=True)
    else:
        combined = res_df
        
    combined.to_csv(os.path.join(TABLE_DIR, "experiment_B_augmented_comparison.csv"), index=False)
    
    # Visualization: Clean-trained vs Augmented-trained
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    for regime, ls, alpha in [("Clean-Only", "--", 0.7), ("Augmented", "-", 1.0)]:
        sub = combined[combined["training_regime"] == regime]
        for m_name in sub["model"].unique():
            m_sub = sub[sub["model"] == m_name]
            label = f"{m_name} ({regime})" if "Augmented" not in m_name else m_name
            ax.plot(
                m_sub["condition"].str.title(),
                m_sub["macro_f1"],
                linestyle=ls,
                marker="o",
                linewidth=2.2,
                label=label,
                alpha=alpha
            )
            
    ax.set_xlabel("Spelling Variation Severity Condition", fontweight="bold")
    ax.set_ylabel("Macro-F1 Score", fontweight="bold")
    ax.set_title("Impact of Robust Data Augmentation on Test-Time Degradation (Experiment B)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_B_augmented_comparison.png"), dpi=300)
    plt.close()
    
    logger.info("Experiment B completed successfully!")
    return combined


if __name__ == "__main__":
    run_experiment_b()
