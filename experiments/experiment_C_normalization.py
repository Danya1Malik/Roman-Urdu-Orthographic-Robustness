"""
Experiment C: Orthographic Normalization Ablation
Ablates the impact of research-specific orthographic normalization versus generic informal NLP preprocessing.
Directly addresses RQ2 (Generic vs Lexicon normalization efficacy) and RQ3 (Variant density predictability).
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalizer import RomanUrduNormalizer
from src.evaluate import compute_classification_metrics
from src.statistical_tests import paired_bootstrap_f1_test
from experiments.experiment_A_zeroshot import predict_transformer
import joblib

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed"
RESULTS_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results"
CHECKPOINT_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\checkpoints"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


def run_experiment_c():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    normalizer = RomanUrduNormalizer()
    
    # Load trained models
    logger.info("Loading pre-trained models from checkpoints...")
    tfidf_path = os.path.join(CHECKPOINT_DIR, "tfidf_baseline.joblib")
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError("TF-IDF checkpoint not found. Please run Experiment A first.")
    tfidf_model = joblib.load(tfidf_path)
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    
    lora_dir = os.path.join(CHECKPOINT_DIR, "lora_xlm_roberta")
    if not os.path.exists(lora_dir):
        raise FileNotFoundError("XLM-R LoRA checkpoint not found. Please run Experiment A first.")
        
    lora_tok = AutoTokenizer.from_pretrained(lora_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)
    lora_model = PeftModel.from_pretrained(base_model, lora_dir).half().to(device)
    
    models = {
        "TF-IDF + LR": lambda texts: tfidf_model.predict([str(t) for t in texts]),
        "XLM-RoBERTa (LoRA)": lambda texts: predict_transformer(lora_model, lora_tok, texts, device)
    }
    
    # Load Severe test set (where degradation is maximal)
    severe_test_df = pd.read_csv(os.path.join(DATA_DIR, "test_severe.csv"))
    clean_test_df = pd.read_csv(os.path.join(DATA_DIR, "test_clean.csv"))
    y_true = severe_test_df["sentiment"].tolist()
    
    normalization_conditions = [
        ("Condition A: Raw (No Norm)", "raw"),
        ("Condition B: Generic Preprocessing", "generic"),
        ("Condition C: Lexicon Normalization", "lexicon"),
        ("Condition D: Combined (Generic + Lexicon)", "combined")
    ]
    
    results = []
    predictions_map = {}
    
    for cond_label, cond_key in normalization_conditions:
        logger.info(f"Applying normalization '{cond_label}' to severe test set...")
        norm_texts = [normalizer.normalize(str(t), condition=cond_key) for t in severe_test_df["review"]]
        
        for m_name, pred_fn in models.items():
            y_pred = pred_fn(norm_texts)
            predictions_map[(m_name, cond_key)] = y_pred
            metrics = compute_classification_metrics(y_true, y_pred)
            results.append({
                "model": m_name,
                "normalization_condition": cond_label,
                "norm_key": cond_key,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"]
            })
            
    res_df = pd.DataFrame(results)
    
    # Calculate recovery metrics relative to raw severe and clean baseline
    # Fetch clean baseline F1
    for m_name in models.keys():
        clean_preds = models[m_name](clean_test_df["review"].tolist())
        clean_f1 = compute_classification_metrics(clean_test_df["sentiment"].tolist(), clean_preds)["macro_f1"]
        raw_f1 = res_df[(res_df["model"] == m_name) & (res_df["norm_key"] == "raw")]["macro_f1"].values[0]
        
        mask = res_df["model"] == m_name
        res_df.loc[mask, "clean_macro_f1"] = clean_f1
        res_df.loc[mask, "raw_severe_f1"] = raw_f1
        res_df.loc[mask, "f1_gain_over_raw"] = res_df.loc[mask, "macro_f1"] - raw_f1
        res_df.loc[mask, "pct_degradation_recovered"] = ((res_df.loc[mask, "macro_f1"] - raw_f1) / (clean_f1 - raw_f1) * 100).clip(lower=0.0)
        
    res_df.to_csv(os.path.join(TABLE_DIR, "experiment_C_normalization_results.csv"), index=False)
    logger.info(f"Normalization Results:\n{res_df[['model', 'normalization_condition', 'macro_f1', 'f1_gain_over_raw', 'pct_degradation_recovered']].to_string(index=False)}")
    
    # -------------------------------------------------------------
    # Statistical Significance Testing (RQ2)
    # -------------------------------------------------------------
    stat_results = {}
    for m_name in models.keys():
        # Condition B (Generic) vs Condition C (Lexicon)
        stat_results[f"{m_name}_generic_vs_lexicon"] = paired_bootstrap_f1_test(
            y_true,
            predictions_map[(m_name, "lexicon")],
            predictions_map[(m_name, "generic")]
        )
        # Condition A (Raw) vs Condition D (Combined)
        stat_results[f"{m_name}_raw_vs_combined"] = paired_bootstrap_f1_test(
            y_true,
            predictions_map[(m_name, "combined")],
            predictions_map[(m_name, "raw")]
        )
        
    with open(os.path.join(TABLE_DIR, "experiment_C_significance.json"), "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2)
        
    # -------------------------------------------------------------
    # RQ3: Density Analysis (Performance degradation vs Variant Density)
    # -------------------------------------------------------------
    logger.info("Computing degradation across variant density quartiles (RQ3)...")
    bins = pd.qcut(severe_test_df["actual_rate"].rank(method="first"), q=4, labels=["Q1 (Low Density)", "Q2 (Medium-Low)", "Q3 (Medium-High)", "Q4 (High Density)"])
    severe_test_df["density_bin"] = bins
    
    density_records = []
    for bin_name in ["Q1 (Low Density)", "Q2 (Medium-Low)", "Q3 (Medium-High)", "Q4 (High Density)"]:
        bin_df = severe_test_df[severe_test_df["density_bin"] == bin_name]
        bin_true = bin_df["sentiment"].tolist()
        
        # Raw severe predictions on bin
        bin_raw_preds = models["XLM-RoBERTa (LoRA)"](bin_df["review"].tolist())
        # Normalized combined predictions on bin
        bin_norm_texts = [normalizer.normalize(str(t), condition="combined") for t in bin_df["review"]]
        bin_norm_preds = models["XLM-RoBERTa (LoRA)"](bin_norm_texts)
        # Clean original predictions on bin
        bin_clean_preds = models["XLM-RoBERTa (LoRA)"](bin_df["review_original"].tolist())
        
        f1_clean = compute_classification_metrics(bin_true, bin_clean_preds)["macro_f1"]
        f1_raw = compute_classification_metrics(bin_true, bin_raw_preds)["macro_f1"]
        f1_norm = compute_classification_metrics(bin_true, bin_norm_preds)["macro_f1"]
        
        density_records.append({
            "density_quartile": bin_name,
            "mean_perturbation_rate": float(bin_df["actual_rate"].mean()),
            "count": len(bin_df),
            "macro_f1_clean": f1_clean,
            "macro_f1_raw_severe": f1_raw,
            "macro_f1_normalized": f1_norm,
            "drop_clean_to_raw": f1_clean - f1_raw,
            "gain_raw_to_norm": f1_norm - f1_raw
        })
        
    density_df = pd.DataFrame(density_records)
    density_df.to_csv(os.path.join(TABLE_DIR, "experiment_C_density_analysis.csv"), index=False)
    
    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    # 1. Normalization Recovery Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cond_labels = ["A: Raw", "B: Generic", "C: Lexicon", "D: Combined"]
    x = np.arange(len(cond_labels))
    width = 0.35
    
    m1_vals = res_df[res_df["model"] == "TF-IDF + LR"]["macro_f1"].values
    m2_vals = res_df[res_df["model"] == "XLM-RoBERTa (LoRA)"]["macro_f1"].values
    
    bars1 = ax.bar(x - width/2, m1_vals, width, label="TF-IDF + LR", color="#7f7f7f", alpha=0.85, edgecolor="black")
    bars2 = ax.bar(x + width/2, m2_vals, width, label="XLM-RoBERTa (LoRA)", color="#2ca02c", alpha=0.85, edgecolor="black")
    
    ax.set_ylabel("Macro-F1 Score Under Severe Perturbation", fontweight="bold")
    ax.set_title("Efficacy of Preprocessing vs Lexicon Normalization on Severe Degradation (RQ2)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontweight="bold")
    ax.legend(frameon=True)
    ax.set_ylim(0, max(m2_vals) * 1.25)
    
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_C_normalization_recovery.png"), dpi=300)
    plt.close()
    
    # 2. Degradation & Recovery vs Variant Density (RQ3)
    fig, ax = plt.subplots(figsize=(9, 5))
    x_bins = range(len(density_df))
    ax.plot(x_bins, density_df["macro_f1_clean"], marker="o", color="#1f77b4", label="Clean Review Text", linewidth=2.2)
    ax.plot(x_bins, density_df["macro_f1_raw_severe"], marker="x", color="#d62728", label="Severe Perturbation (Unnormalized)", linewidth=2.2, linestyle="--")
    ax.plot(x_bins, density_df["macro_f1_normalized"], marker="s", color="#2ca02c", label="Severe + Combined Normalization", linewidth=2.2)
    
    ax.set_xticks(x_bins)
    ax.set_xticklabels(density_df["density_quartile"], fontweight="bold")
    ax.set_xlabel("Review Spelling Variant Density Quartile", fontweight="bold")
    ax.set_ylabel("Macro-F1 Score", fontweight="bold")
    ax.set_title("Degradation and Normalization Recovery Across Variant Density (RQ3)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_C_density_vs_recovery.png"), dpi=300)
    plt.close()
    
    logger.info("Experiment C completed successfully!")
    return res_df, density_df, stat_results


if __name__ == "__main__":
    run_experiment_c()
