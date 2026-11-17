"""
Tokenization Analysis Module: Subword Fragmentation & Fertility
Compares mBERT (WordPiece) and XLM-RoBERTa (SentencePiece/BPE) on Roman Urdu Common vs Spelling Variants
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLEAN_LEXICON_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_lexicon.csv"
FIG_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\figures"
TABLE_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables"


def run_tokenization_analysis():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    
    logger.info("Loading tokenizers: bert-base-multilingual-cased and xlm-roberta-base...")
    tok_mbert = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    tok_xlmr = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    logger.info("Loading cleaned lexicon...")
    lex = pd.read_csv(CLEAN_LEXICON_PATH, encoding="utf-8")
    var_cols = ["var-1", "var-2", "var-3", "var-4", "var-5"]
    
    results = []
    
    for idx, row in lex.iterrows():
        common = str(row['Common']).strip()
        if not common or common == "nan":
            continue
            
        # Common word subwords (excluding special tokens)
        sub_mbert_common = tok_mbert.tokenize(common)
        sub_xlmr_common = tok_xlmr.tokenize(common)
        
        len_mbert_common = len(sub_mbert_common)
        len_xlmr_common = len(sub_xlmr_common)
        
        for col in var_cols:
            variant_val = row[col]
            if pd.notnull(variant_val):
                variant = str(variant_val).strip()
                if variant and variant != "nan" and variant.lower() != common.lower():
                    sub_mbert_var = tok_mbert.tokenize(variant)
                    sub_xlmr_var = tok_xlmr.tokenize(variant)
                    
                    len_mbert_var = len(sub_mbert_var)
                    len_xlmr_var = len(sub_xlmr_var)
                    
                    results.append({
                        "common": common,
                        "variant": variant,
                        "mbert_common_subwords": len_mbert_common,
                        "mbert_variant_subwords": len_mbert_var,
                        "mbert_fertility_increase": len_mbert_var - len_mbert_common,
                        "xlmr_common_subwords": len_xlmr_common,
                        "xlmr_variant_subwords": len_xlmr_var,
                        "xlmr_fertility_increase": len_xlmr_var - len_xlmr_common
                    })
                    
    df_tok = pd.DataFrame(results)
    df_tok.to_csv(os.path.join(TABLE_DIR, "tokenization_fertility_pairs.csv"), index=False)
    
    summary = {
        "total_pairs_analyzed": len(df_tok),
        "mbert": {
            "mean_common_subwords": float(df_tok['mbert_common_subwords'].mean()),
            "mean_variant_subwords": float(df_tok['mbert_variant_subwords'].mean()),
            "pct_fragmentation_increase": float((df_tok['mbert_fertility_increase'] > 0).mean() * 100),
            "pct_unchanged": float((df_tok['mbert_fertility_increase'] == 0).mean() * 100),
            "pct_decreased": float((df_tok['mbert_fertility_increase'] < 0).mean() * 100),
            "mean_increase": float(df_tok['mbert_fertility_increase'].mean())
        },
        "xlmr": {
            "mean_common_subwords": float(df_tok['xlmr_common_subwords'].mean()),
            "mean_variant_subwords": float(df_tok['xlmr_variant_subwords'].mean()),
            "pct_fragmentation_increase": float((df_tok['xlmr_fertility_increase'] > 0).mean() * 100),
            "pct_unchanged": float((df_tok['xlmr_fertility_increase'] == 0).mean() * 100),
            "pct_decreased": float((df_tok['xlmr_fertility_increase'] < 0).mean() * 100),
            "mean_increase": float(df_tok['xlmr_fertility_increase'].mean())
        }
    }
    
    with open(os.path.join(TABLE_DIR, "tokenization_fragmentation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Tokenization analysis summary:\n{json.dumps(summary, indent=2)}")
    
    # Plotting Fragmentation Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Subword fertility comparison
    models = ['mBERT\n(WordPiece)', 'XLM-RoBERTa\n(SentencePiece)']
    x = np.arange(len(models))
    width = 0.35
    
    common_means = [summary['mbert']['mean_common_subwords'], summary['xlmr']['mean_common_subwords']]
    variant_means = [summary['mbert']['mean_variant_subwords'], summary['xlmr']['mean_variant_subwords']]
    
    bars1 = axes[0].bar(x - width/2, common_means, width, label='Canonical Common', color='#4575b4', alpha=0.9)
    bars2 = axes[0].bar(x + width/2, variant_means, width, label='Spelling Variant', color='#d73027', alpha=0.9)
    
    axes[0].set_ylabel('Mean Subwords per Word (Fertility)', fontweight='bold')
    axes[0].set_title('Subword Fragmentation: Common vs Spelling Variants', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, max(variant_means) * 1.3)
    
    for bar in bars1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}", ha='center', fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}", ha='center', fontweight='bold')
        
    # Right: Distribution of Fertility Change
    bins = np.arange(-2.5, 4.5, 1)
    axes[1].hist([df_tok['mbert_fertility_increase'], df_tok['xlmr_fertility_increase']], 
                bins=bins, label=['mBERT Change', 'XLM-R Change'], color=['#4575b4', '#d73027'], alpha=0.8)
    axes[1].set_xlabel('Change in Subword Tokens (Variant Subwords - Common Subwords)', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Fertility Shift Distribution Under Spelling Variation', fontweight='bold')
    axes[1].legend()
    axes[1].set_xticks(range(-2, 4))
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tokenizer_fragmentation_comparison.png"), dpi=300)
    plt.close()
    
    logger.info("Saved tokenizer fragmentation analysis figure and tables.")
    return summary


if __name__ == "__main__":
    run_tokenization_analysis()
