"""
Evaluation and Metrics Computation Module
Computes:
  - Accuracy, Macro-F1, Precision, Recall, Per-Class Metrics
  - Confusion Matrices & Heatmap Visualizations
  - Degradation Quantification (Delta and Relative Percentage Drop)
  - 3-Class Collapsed Evaluation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLASS_NAMES_5 = ["very negative", "negative", "neutral", "positive", "very positive"]
CLASS_NAMES_3 = ["negative", "neutral", "positive"]

CM_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\confusion_matrices"
TABLE_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables"
FIG_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\figures"


def compute_classification_metrics(y_true, y_pred, labels=CLASS_NAMES_5):
    """
    Computes standard 5-class evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    per_class_metrics = {}
    for i, c in enumerate(labels):
        per_class_metrics[c] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }


def compute_3class_metrics(y_true_5, y_pred_5):
    """
    Collapse 5-class predictions to 3-class (for external RUECD comparison):
    very positive / positive -> positive
    very negative / negative -> negative
    neutral -> neutral
    """
    mapping = {
        "very positive": "positive",
        "positive": "positive",
        "neutral": "neutral",
        "negative": "negative",
        "very negative": "negative"
    }
    y_true_3 = [mapping.get(y, y) for y in y_true_5]
    y_pred_3 = [mapping.get(y, y) for y in y_pred_5]
    
    return compute_classification_metrics(y_true_3, y_pred_3, labels=CLASS_NAMES_3)


def plot_and_save_confusion_matrix(cm_list, labels, title, filename):
    os.makedirs(CM_DIR, exist_ok=True)
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[l.title() for l in labels],
        yticklabels=[l.title() for l in labels],
        ax=ax,
        cbar=True
    )
    ax.set_xlabel("Predicted Sentiment", fontweight="bold")
    ax.set_ylabel("True Sentiment", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    plt.tight_layout()
    
    save_path = os.path.join(CM_DIR, filename)
    fig.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def compute_degradation(clean_metric, perturbed_metric):
    """
    Computes absolute degradation and relative percentage drop.
    """
    delta = clean_metric - perturbed_metric
    rel_drop = (delta / clean_metric * 100.0) if clean_metric > 0 else 0.0
    return {"delta": float(delta), "relative_pct_drop": float(rel_drop)}
