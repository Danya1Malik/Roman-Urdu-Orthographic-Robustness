"""
Statistical Significance Testing Module
Implements:
  1. Paired Bootstrap Resampling for Macro-F1 differences (95% CI & p-value)
  2. McNemar's Test for paired prediction discordance
  3. Two-Way ANOVA for Factorial Grid analysis (Rate x Edit Distance)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
import statsmodels.api as sm
from statsmodels.formula.api import ols

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TABLE_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables"


def paired_bootstrap_f1_test(y_true, y_pred_a, y_pred_b, n_bootstraps=1000, seed=42):
    """
    Paired bootstrap test to evaluate whether Model A is significantly different from Model B in Macro-F1.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1_a_orig = f1_score(y_true, y_pred_a, average='macro', zero_division=0)
    f1_b_orig = f1_score(y_true, y_pred_b, average='macro', zero_division=0)
    delta_orig = f1_a_orig - f1_b_orig
    
    boot_deltas = []
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        b_true = y_true[indices]
        b_pred_a = y_pred_a[indices]
        b_pred_b = y_pred_b[indices]
        
        f1_a = f1_score(b_true, b_pred_a, average='macro', zero_division=0)
        f1_b = f1_score(b_true, b_pred_b, average='macro', zero_division=0)
        boot_deltas.append(f1_a - f1_b)
        
    boot_deltas = np.array(boot_deltas)
    ci_lower = np.percentile(boot_deltas, 2.5)
    ci_upper = np.percentile(boot_deltas, 97.5)
    
    # Two-tailed empirical p-value under null hypothesis (delta = 0)
    # Centered bootstrap distribution
    centered_deltas = boot_deltas - delta_orig
    p_value = np.mean(np.abs(centered_deltas) >= np.abs(delta_orig))
    
    return {
        "f1_a": float(f1_a_orig),
        "f1_b": float(f1_b_orig),
        "delta": float(delta_orig),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant_p05": bool(p_value < 0.05),
        "significant_p01": bool(p_value < 0.01)
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test with continuity correction on paired binary classification correctness.
    """
    y_true = np.array(y_true)
    corr_a = (np.array(y_pred_a) == y_true)
    corr_b = (np.array(y_pred_b) == y_true)
    
    # 2x2 contingency table:
    # n00: both wrong, n01: A wrong B correct, n10: A correct B wrong, n11: both correct
    n01 = np.sum((~corr_a) & corr_b)  # discordant: B correct, A incorrect
    n10 = np.sum(corr_a & (~corr_b))  # discordant: A correct, B incorrect
    
    if (n01 + n10) == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": int(n01), "n10": int(n10)}
        
    # Edwards continuity correction
    chi2 = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    p_val = stats.chi2.sf(chi2, df=1)
    
    return {
        "chi2": float(chi2),
        "p_value": float(p_val),
        "n01": int(n01),
        "n10": int(n10),
        "significant": bool(p_val < 0.05)
    }


def two_way_anova_factorial(results_df):
    """
    Runs Two-Way ANOVA on Factorial Grid results:
    DV: accuracy or macro_f1
    IVs: rate (categorical: 25%, 50%, 90%), distance_strategy (categorical: low, random, high)
    """
    try:
        model = ols('macro_f1 ~ C(rate) + C(strategy) + C(rate):C(strategy)', data=results_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table.to_dict()
    except Exception as e:
        logger.warning(f"Two-Way ANOVA computation note: {e}")
        return {}
