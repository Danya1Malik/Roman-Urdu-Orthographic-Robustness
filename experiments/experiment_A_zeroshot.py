"""
Experiment A: Zero-Shot Robustness Evaluation
Evaluates TF-IDF Baseline, mBERT, XLM-RoBERTa Full, and XLM-RoBERTa LoRA
trained strictly on clean data, across:
  1. Canonical severity conditions: Clean, Mild, Moderate, Severe
  2. 3x3 Factorial Grid: Rate {25%, 50%, 90%} x Edit Distance {Low, Random, High}
Addresses RQ1 and RQ4.
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    build_tfidf_baseline,
    get_transformer_model,
    get_transformer_tokenizer,
    LABEL_TO_ID,
    ID_TO_LABEL
)
from src.train import (
    train_tfidf,
    train_transformer,
    TextClassificationDataset,
    CHECKPOINT_DIR
)
from src.evaluate import (
    compute_classification_metrics,
    plot_and_save_confusion_matrix,
    compute_degradation,
    CLASS_NAMES_5
)
from src.statistical_tests import (
    paired_bootstrap_f1_test,
    mcnemar_test
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed"
RESULTS_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


def predict_transformer(model, tokenizer, texts, device, batch_size=64, max_length=64):
    model.eval()
    dataset = TextClassificationDataset(texts, labels=None, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend([ID_TO_LABEL[p] for p in batch_preds])
    return preds


def run_experiment_a():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    
    # Load splits
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    
        # -------------------------------------------------------------
    # 1. Train or Load Models
    # -------------------------------------------------------------
    models_dict = {}
    import joblib
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    
    # A. TF-IDF + Logistic Regression
    tfidf_path = os.path.join(CHECKPOINT_DIR, "tfidf_baseline.joblib")
    if os.path.exists(tfidf_path):
        logger.info(f"Loading existing TF-IDF Baseline from {tfidf_path}")
        tfidf_model = joblib.load(tfidf_path)
    else:
        logger.info("=== 1. Training TF-IDF Baseline ===")
        tfidf_model, tfidf_val_metrics = train_tfidf(train_df, val_df, save_name="tfidf_baseline.joblib")
    models_dict["TF-IDF + LR"] = {
        "type": "sklearn",
        "model": tfidf_model,
        "predict_fn": lambda texts: tfidf_model.predict([str(t) for t in texts])
    }
    
    # B. mBERT Full Fine-Tuning
    mbert_dir = os.path.join(CHECKPOINT_DIR, "full_mbert")
    if os.path.exists(os.path.join(mbert_dir, "model.safetensors")):
        logger.info(f"Loading existing mBERT from {mbert_dir}")
        mbert_tok = AutoTokenizer.from_pretrained(mbert_dir)
        mbert_model = AutoModelForSequenceClassification.from_pretrained(mbert_dir).to(device)
    else:
        logger.info("=== 2. Training mBERT Full Fine-Tuning ===")
        mbert_model, mbert_tok, mbert_val = train_transformer(
            model_name="bert-base-multilingual-cased",
            use_lora=False,
            train_df=train_df,
            val_df=val_df,
            epochs=2,
            batch_size=32,
            lr=3e-5,
            max_length=64,
            save_name="full_mbert",
            device=device
        )
    models_dict["mBERT (Full)"] = {
        "type": "torch",
        "model": mbert_model,
        "tokenizer": mbert_tok,
        "predict_fn": lambda texts: predict_transformer(mbert_model, mbert_tok, texts, device)
    }
    
    # C. XLM-RoBERTa Full Fine-Tuning
    xlmr_dir = os.path.join(CHECKPOINT_DIR, "full_xlm_roberta")
    if os.path.exists(os.path.join(xlmr_dir, "model.safetensors")):
        logger.info(f"Loading existing XLM-RoBERTa (Full) from {xlmr_dir}")
        xlmr_tok = AutoTokenizer.from_pretrained(xlmr_dir)
        xlmr_model = AutoModelForSequenceClassification.from_pretrained(xlmr_dir).to(device)
    else:
        logger.info("=== 3. Training XLM-RoBERTa Full Fine-Tuning ===")
        xlmr_model, xlmr_tok, xlmr_val = train_transformer(
            model_name="xlm-roberta-base",
            use_lora=False,
            train_df=train_df,
            val_df=val_df,
            epochs=2,
            batch_size=32,
            lr=2e-5,
            max_length=64,
            save_name="full_xlm_roberta",
            device=device
        )
    models_dict["XLM-RoBERTa (Full)"] = {
        "type": "torch",
        "model": xlmr_model,
        "tokenizer": xlmr_tok,
        "predict_fn": lambda texts: predict_transformer(xlmr_model, xlmr_tok, texts, device)
    }
    
    # D. XLM-RoBERTa LoRA Adaptation
    lora_dir = os.path.join(CHECKPOINT_DIR, "lora_xlm_roberta")
    if os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(lora_dir, "adapter_model.bin")):
        logger.info(f"Loading existing XLM-RoBERTa (LoRA) from {lora_dir}")
        lora_tok = AutoTokenizer.from_pretrained(lora_dir)
        base_m = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)
        lora_model = PeftModel.from_pretrained(base_m, lora_dir).to(device)
    else:
        logger.info("=== 4. Training XLM-RoBERTa LoRA Adaptation ===")
        lora_model, lora_tok, lora_val = train_transformer(
            model_name="xlm-roberta-base",
            use_lora=True,
            train_df=train_df,
            val_df=val_df,
            epochs=3,
            batch_size=32,
            lr=2e-4,
            max_length=64,
            save_name="lora_xlm_roberta",
            device=device
        )
    models_dict["XLM-RoBERTa (LoRA)"] = {
        "type": "torch",
        "model": lora_model,
        "tokenizer": lora_tok,
        "predict_fn": lambda texts: predict_transformer(lora_model, lora_tok, texts, device)
    }
    
    # -------------------------------------------------------------
    # 2. Zero-Shot Evaluation on Canonical Severity Conditions
    # -------------------------------------------------------------
    logger.info("=== 5. Evaluating Models Across Severity Conditions ===")
    canonical_conditions = ["clean", "mild", "moderate", "severe"]
    
    records = []
    predictions_store = {}
    
    for cond in canonical_conditions:
        test_file = os.path.join(DATA_DIR, f"test_{cond}.csv")
        test_df = pd.read_csv(test_file)
        y_true = test_df["sentiment"].tolist()
        
        for model_name, m_info in models_dict.items():
            logger.info(f"Evaluating {model_name} on condition: {cond}...")
            y_pred = m_info["predict_fn"](test_df["review"].tolist())
            predictions_store[(model_name, cond)] = y_pred
            
            metrics = compute_classification_metrics(y_true, y_pred)
            
            # Save confusion matrix heatmap
            cm_filename = f"cm_{model_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').lower()}_{cond}.png"
            plot_and_save_confusion_matrix(
                metrics["confusion_matrix"],
                CLASS_NAMES_5,
                f"Confusion Matrix: {model_name} [{cond.title()} Condition]",
                cm_filename
            )
            
            records.append({
                "model": model_name,
                "condition": cond,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "weighted_f1": metrics["weighted_f1"]
            })
            
    res_df = pd.DataFrame(records)
    
    # Compute degradation deltas relative to Clean
    for model_name in models_dict.keys():
        clean_row = res_df[(res_df["model"] == model_name) & (res_df["condition"] == "clean")].iloc[0]
        clean_f1 = clean_row["macro_f1"]
        clean_acc = clean_row["accuracy"]
        
        for cond in canonical_conditions:
            mask = (res_df["model"] == model_name) & (res_df["condition"] == cond)
            curr_f1 = res_df.loc[mask, "macro_f1"].values[0]
            curr_acc = res_df.loc[mask, "accuracy"].values[0]
            
            res_df.loc[mask, "delta_macro_f1"] = clean_f1 - curr_f1
            res_df.loc[mask, "rel_drop_macro_f1_pct"] = ((clean_f1 - curr_f1) / clean_f1) * 100
            res_df.loc[mask, "delta_accuracy"] = clean_acc - curr_acc
            res_df.loc[mask, "rel_drop_accuracy_pct"] = ((clean_acc - curr_acc) / clean_acc) * 100
            
    res_df.to_csv(os.path.join(TABLE_DIR, "experiment_A_canonical_results.csv"), index=False)
    logger.info(f"Canonical Results Table:\n{res_df[['model', 'condition', 'accuracy', 'macro_f1', 'delta_macro_f1', 'rel_drop_macro_f1_pct']].to_string(index=False)}")
    
    # -------------------------------------------------------------
    # 3. Factorial Grid Evaluation (3 rates x 3 edit distances)
    # -------------------------------------------------------------
    logger.info("=== 6. Evaluating Models Across 3x3 Factorial Grid ===")
    factorial_rates = [25, 50, 90]
    factorial_strats = ["low", "random", "high"]
    grid_records = []
    
    for r in factorial_rates:
        for s in factorial_strats:
            grid_cond = f"grid_rate{r}_{s}"
            test_file = os.path.join(DATA_DIR, f"test_{grid_cond}.csv")
            if not os.path.exists(test_file):
                continue
            test_df = pd.read_csv(test_file)
            y_true = test_df["sentiment"].tolist()
            
            for model_name, m_info in models_dict.items():
                y_pred = m_info["predict_fn"](test_df["review"].tolist())
                metrics = compute_classification_metrics(y_true, y_pred)
                grid_records.append({
                    "model": model_name,
                    "rate": r,
                    "strategy": s,
                    "condition": grid_cond,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"]
                })
                
    grid_df = pd.DataFrame(grid_records)
    grid_df.to_csv(os.path.join(TABLE_DIR, "experiment_A_factorial_grid.csv"), index=False)
    
    # -------------------------------------------------------------
    # 4. Statistical Significance Testing
    # -------------------------------------------------------------
    logger.info("=== 7. Running Statistical Significance Tests ===")
    clean_test_df = pd.read_csv(os.path.join(DATA_DIR, "test_clean.csv"))
    y_true_clean = clean_test_df["sentiment"].tolist()
    severe_test_df = pd.read_csv(os.path.join(DATA_DIR, "test_severe.csv"))
    y_true_severe = severe_test_df["sentiment"].tolist()
    
    significance_results = {}
    
    # Comparison 1: XLM-R Full vs LoRA on Clean
    significance_results["xlmr_full_vs_lora_clean"] = paired_bootstrap_f1_test(
        y_true_clean,
        predictions_store[("XLM-RoBERTa (Full)", "clean")],
        predictions_store[("XLM-RoBERTa (LoRA)", "clean")]
    )
    
    # Comparison 2: XLM-R Full vs LoRA on Severe (RQ4)
    significance_results["xlmr_full_vs_lora_severe"] = paired_bootstrap_f1_test(
        y_true_severe,
        predictions_store[("XLM-RoBERTa (Full)", "severe")],
        predictions_store[("XLM-RoBERTa (LoRA)", "severe")]
    )
    
    # Comparison 3: McNemar Test for XLM-R Full vs LoRA on Severe
    significance_results["mcnemar_xlmr_full_vs_lora_severe"] = mcnemar_test(
        y_true_severe,
        predictions_store[("XLM-RoBERTa (Full)", "severe")],
        predictions_store[("XLM-RoBERTa (LoRA)", "severe")]
    )
    
    # Comparison 4: XLM-R Full Clean vs Severe Degradation
    significance_results["xlmr_full_clean_vs_severe"] = paired_bootstrap_f1_test(
        y_true_severe,
        predictions_store[("XLM-RoBERTa (Full)", "clean")],
        predictions_store[("XLM-RoBERTa (Full)", "severe")]
    )
    
    # Comparison 5: XLM-R LoRA Clean vs Severe Degradation
    significance_results["xlmr_lora_clean_vs_severe"] = paired_bootstrap_f1_test(
        y_true_severe,
        predictions_store[("XLM-RoBERTa (LoRA)", "clean")],
        predictions_store[("XLM-RoBERTa (LoRA)", "severe")]
    )
    
    with open(os.path.join(TABLE_DIR, "experiment_A_significance.json"), "w", encoding="utf-8") as f:
        json.dump(significance_results, f, indent=2)
        
    logger.info(f"Significance Summary:\n{json.dumps(significance_results, indent=2)}")
    
    # -------------------------------------------------------------
    # 5. Visualizations
    # -------------------------------------------------------------
    logger.info("=== 8. Generating Visualizations ===")
    
    # Robustness Degradation Curves (Macro-F1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {
        "TF-IDF + LR": "#7f7f7f",
        "mBERT (Full)": "#ff7f0e",
        "XLM-RoBERTa (Full)": "#1f77b4",
        "XLM-RoBERTa (LoRA)": "#2ca02c"
    }
    markers = {
        "TF-IDF + LR": "s",
        "mBERT (Full)": "^",
        "XLM-RoBERTa (Full)": "o",
        "XLM-RoBERTa (LoRA)": "D"
    }
    
    for model_name in models_dict.keys():
        sub = res_df[res_df["model"] == model_name]
        ax.plot(
            sub["condition"].str.title(),
            sub["macro_f1"],
            label=model_name,
            color=palette.get(model_name, "black"),
            marker=markers.get(model_name, "o"),
            linewidth=2.5,
            markersize=8
        )
        
    ax.set_xlabel("Spelling Variation Severity Condition", fontweight="bold", fontsize=11)
    ax.set_ylabel("Macro-F1 Score", fontweight="bold", fontsize=11)
    ax.set_title("Robustness Degradation Across Spelling Variation Severity (RQ1 & RQ4)", fontweight="bold", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_A_robustness_curves.png"), dpi=300)
    plt.close()
    
    # Relative Degradation Drop Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    severe_drops = res_df[res_df["condition"] == "severe"]
    x = np.arange(len(severe_drops))
    bars = ax.bar(x, severe_drops["rel_drop_macro_f1_pct"], color=[palette.get(m, "blue") for m in severe_drops["model"]], alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(severe_drops["model"], fontweight="bold")
    ax.set_ylabel("Relative Macro-F1 Drop Under Severe Variation (%)", fontweight="bold")
    ax.set_title("Vulnerability to Severe Orthographic Perturbation Across Architectures", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, max(severe_drops["rel_drop_macro_f1_pct"]) * 1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_A_relative_drop.png"), dpi=300)
    plt.close()
    
    # Factorial Heatmap: Rate x Edit Distance for XLM-R Full vs LoRA
    if len(grid_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for idx, m_target in enumerate(["XLM-RoBERTa (Full)", "XLM-RoBERTa (LoRA)"]):
            m_sub = grid_df[grid_df["model"] == m_target]
            pivot_f1 = m_sub.pivot(index="rate", columns="strategy", values="macro_f1")[["low", "random", "high"]]
            sns.heatmap(pivot_f1, annot=True, fmt=".4f", cmap="YlGnBu", ax=axes[idx], cbar=True)
            axes[idx].set_title(f"{m_target}\nMacro-F1 on Factorial Grid", fontweight="bold")
            axes[idx].set_xlabel("Edit Distance Strategy", fontweight="bold")
            axes[idx].set_ylabel("Perturbation Rate (%)", fontweight="bold")
            axes[idx].set_xticklabels(["Low (Min)", "Random", "High (Max)"])
            axes[idx].set_yticklabels(["25%", "50%", "90%"])
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "experiment_A_factorial_heatmap.png"), dpi=300)
        plt.close()
        
    logger.info("Experiment A successfully completed!")
    return res_df, grid_df, significance_results


if __name__ == "__main__":
    run_experiment_a()
