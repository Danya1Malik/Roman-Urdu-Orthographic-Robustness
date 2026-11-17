"""
Master Experiment Runner
Executes Experiment A, Experiment B, Experiment C, and External Validation in sequence.
Consolidates all empirical metrics, tables, and figures.
"""

import os
import sys
import time
import json
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.experiment_A_zeroshot import run_experiment_a
from experiments.experiment_B_augmented import run_experiment_b
from experiments.experiment_C_normalization import run_experiment_c
from src.external_validation import evaluate_model_on_ruecd
import joblib

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHECKPOINT_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\checkpoints"
TABLE_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables"


def run_master_pipeline():
    start_total = time.time()
    logger.info("================================================================")
    logger.info("STARTING MASTER EXPERIMENTAL EVALUATION PIPELINE")
    logger.info("================================================================")
    
    # 1. Run Experiment A: Zero-Shot Robustness
    if os.path.exists(os.path.join(TABLE_DIR, "experiment_A_canonical_results.csv")) and os.path.exists(os.path.join(TABLE_DIR, "experiment_A_factorial_grid.csv")):
        logger.info(">>> STAGE 1: Experiment A already completed. Skipping to next stage.")
    else:
        logger.info(">>> STAGE 1: Running Experiment A (Zero-Shot Robustness & Factorial Grid)...")
        res_a, grid_a, sig_a = run_experiment_a()
    
    # 2. Run Experiment B: Augmented Training
    if os.path.exists(os.path.join(TABLE_DIR, "experiment_B_augmented_comparison.csv")):
        logger.info(">>> STAGE 2: Experiment B already completed. Skipping to next stage.")
    else:
        logger.info(">>> STAGE 2: Running Experiment B (Augmented Training vs Clean)...")
        res_b = run_experiment_b()
    
    # 3. Run Experiment C: Normalization Ablation
    if os.path.exists(os.path.join(TABLE_DIR, "experiment_C_density_analysis.csv")):
        logger.info(">>> STAGE 3: Experiment C already completed. Skipping to next stage.")
    else:
        logger.info(">>> STAGE 3: Running Experiment C (Normalization Ablation & Density)...")
        res_c, density_c, sig_c = run_experiment_c()
    
    # 4. Run External Validation on RUECD
    logger.info(">>> STAGE 4: Running External Validation on RUECD (Daraz.pk reviews)...")
    tfidf_model = joblib.load(os.path.join(CHECKPOINT_DIR, "tfidf_baseline.joblib"))
    ruecd_tfidf_res = evaluate_model_on_ruecd(
        predict_fn=lambda texts: tfidf_model.predict([str(t) for t in texts]),
        model_name="TF-IDF + LR",
        disjoint_only=True
    )
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    import torch
    from experiments.experiment_A_zeroshot import predict_transformer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_dir = os.path.join(CHECKPOINT_DIR, "lora_xlm_roberta")
    lora_tok = AutoTokenizer.from_pretrained(lora_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)
    lora_model = PeftModel.from_pretrained(base_model, lora_dir).half().to(device)
    
    ruecd_lora_res = evaluate_model_on_ruecd(
        predict_fn=lambda texts: predict_transformer(lora_model, lora_tok, texts, device),
        model_name="XLM-RoBERTa (LoRA)",
        disjoint_only=True
    )
    
    master_ruecd = pd.concat([ruecd_tfidf_res, ruecd_lora_res], ignore_index=True)
    master_ruecd.to_csv(os.path.join(TABLE_DIR, "external_validation_ruecd_results.csv"), index=False)
    logger.info(f"External Validation Results:\n{master_ruecd.to_string(index=False)}")
    
    total_time = time.time() - start_total
    logger.info("================================================================")
    logger.info(f"ALL EXPERIMENTS SUCCESSFULLY EXECUTED IN {total_time/60:.2f} MINUTES")
    logger.info("================================================================")
    
    master_manifest = {
        "execution_time_minutes": total_time / 60,
        "completed_experiments": ["Experiment A", "Experiment B", "Experiment C", "External Validation"],
        "tables_generated": [
            "experiment_A_canonical_results.csv",
            "experiment_A_factorial_grid.csv",
            "experiment_B_augmented_comparison.csv",
            "experiment_C_normalization_results.csv",
            "experiment_C_density_analysis.csv",
            "external_validation_ruecd_results.csv"
        ]
    }
    with open(os.path.join(TABLE_DIR, "master_experiments_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)
        
    return master_manifest


if __name__ == "__main__":
    run_master_pipeline()
