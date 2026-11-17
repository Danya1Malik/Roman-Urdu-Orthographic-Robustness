"""
Exploratory Data Analysis (EDA) Module
Project: Robustness of Roman Urdu Sentiment Classification to Orthographic Spelling Variation
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.autolayout'] = True

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLEAN_REVIEWS_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_reviews.csv"
CLEAN_LEXICON_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_lexicon.csv"
RAW_REVIEWS_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\Dataset\Roman Urdu reviews Dataset with English translation.csv"

FIG_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\figures"
TABLE_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables"


def levenshtein_distance(s1, s2):
    """Compute standard Levenshtein edit distance between two strings."""
    s1, s2 = str(s1).lower(), str(s2).lower()
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def run_eda():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    
    logger.info("Loading cleaned datasets for EDA...")
    df_clean = pd.read_csv(CLEAN_REVIEWS_PATH, encoding="utf-8")
    df_raw = pd.read_csv(RAW_REVIEWS_PATH, encoding="latin-1")
    lex = pd.read_csv(CLEAN_LEXICON_PATH, encoding="utf-8")
    
    # -------------------------------------------------------------
    # 1. Class Distribution Analysis (Raw vs Deduplicated)
    # -------------------------------------------------------------
    logger.info("Analyzing class distributions...")
    raw_sentiment_clean = df_raw['SENTIMENT'].astype(str).str.strip().str.lower().replace({'neative': 'negative'})
    raw_counts = raw_sentiment_clean.value_counts()
    clean_counts = df_clean['sentiment'].value_counts()
    
    class_order = ["very negative", "negative", "neutral", "positive", "very positive"]
    dist_df = pd.DataFrame({
        "Class": class_order,
        "Raw Count": [raw_counts.get(c, 0) for c in class_order],
        "Raw %": [(raw_counts.get(c, 0)/len(df_raw)*100) for c in class_order],
        "Clean Count": [clean_counts.get(c, 0) for c in class_order],
        "Clean %": [(clean_counts.get(c, 0)/len(df_clean)*100) for c in class_order]
    })
    dist_df.to_csv(os.path.join(TABLE_DIR, "class_distribution_comparison.csv"), index=False)
    
    # Plot Class Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(class_order))
    width = 0.35
    
    ax.bar(x - width/2, dist_df["Raw Count"], width, label=f"Raw ({len(df_raw):,} rows)", color="#4c72b0", alpha=0.85)
    ax.bar(x + width/2, dist_df["Clean Count"], width, label=f"Deduplicated ({len(df_clean):,} rows)", color="#55a868", alpha=0.85)
    
    ax.set_xlabel("Sentiment Class", fontweight="bold")
    ax.set_ylabel("Number of Reviews", fontweight="bold")
    ax.set_title("Sentiment Class Distribution: Raw vs Deduplicated RUWV-NSR", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in class_order])
    ax.legend(frameon=True)
    
    # Add text labels on bars
    for i, v in enumerate(dist_df["Raw Count"]):
        ax.text(i - width/2, v + 80, f"{v:,}\n({dist_df['Raw %'][i]:.1f}%)", ha="center", fontsize=8)
    for i, v in enumerate(dist_df["Clean Count"]):
        ax.text(i + width/2, v + 80, f"{v:,}\n({dist_df['Clean %'][i]:.1f}%)", ha="center", fontsize=8)
        
    ax.set_ylim(0, max(dist_df["Raw Count"]) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Review Length & Lexical Statistics
    # -------------------------------------------------------------
    logger.info("Computing review length and lexical statistics...")
    df_clean['char_length'] = df_clean['review'].astype(str).str.len()
    df_clean['tokens'] = df_clean['review'].astype(str).apply(lambda r: re.findall(r'\b[a-zA-Z]+\b', r.lower()))
    df_clean['word_count'] = df_clean['tokens'].apply(len)
    df_clean['unique_words'] = df_clean['tokens'].apply(lambda t: len(set(t)))
    df_clean['ttr'] = df_clean.apply(lambda row: row['unique_words'] / row['word_count'] if row['word_count'] > 0 else 0, axis=1)

    # Length statistics overall and per class
    length_summary = df_clean.groupby('sentiment')['word_count'].agg(
        Mean='mean',
        Std='std',
        Median='median',
        Q25=lambda x: x.quantile(0.25),
        Q75=lambda x: x.quantile(0.75),
        Q95=lambda x: x.quantile(0.95),
        Max='max'
    ).loc[class_order]
    length_summary.round(2).to_csv(os.path.join(TABLE_DIR, "review_length_by_sentiment.csv"))

    # Plot Review Length Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram with 95th percentile
    p95 = df_clean['word_count'].quantile(0.95)
    sns.histplot(df_clean['word_count'], bins=60, kde=True, ax=axes[0], color='#2b5c8f')
    axes[0].axvline(df_clean['word_count'].mean(), color='red', linestyle='--', label=f"Mean: {df_clean['word_count'].mean():.1f}")
    axes[0].axvline(df_clean['word_count'].median(), color='green', linestyle=':', label=f"Median: {df_clean['word_count'].median():.0f}")
    axes[0].axvline(p95, color='orange', linestyle='-.', label=f"95th Pct: {p95:.0f}")
    axes[0].set_xlim(0, 80)
    axes[0].set_xlabel("Word Count per Review", fontweight="bold")
    axes[0].set_ylabel("Frequency", fontweight="bold")
    axes[0].set_title("Distribution of Review Word Counts", fontweight="bold")
    axes[0].legend()
    
    # Right: Boxplot by class
    sns.boxplot(data=df_clean, x='sentiment', y='word_count', order=class_order, ax=axes[1], palette='Blues_r', showfliers=False)
    axes[1].set_xlabel("Sentiment Class", fontweight="bold")
    axes[1].set_ylabel("Word Count (excluding fliers)", fontweight="bold")
    axes[1].set_xticklabels([c.title() for c in class_order])
    axes[1].set_title("Review Word Count by Sentiment Class", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "review_length_distribution.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Lexicon Structure and Edit Distance Statistics
    # -------------------------------------------------------------
    logger.info("Computing Lexicon variation and edit distance statistics...")
    var_cols = ["var-1", "var-2", "var-3", "var-4", "var-5"]
    
    # Variant count distribution
    var_counts = lex['num_recorded_variants'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(var_counts.index, var_counts.values, color='#4575b4', edgecolor='black', alpha=0.85)
    ax.set_xlabel("Number of Documented Spelling Variants per Headword", fontweight="bold")
    ax.set_ylabel("Number of Headwords", fontweight="bold")
    ax.set_title("Distribution of Documented Spelling Variants in RUWV-NSR Lexicon (N=5,219)", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 25, f"{h:,}\n({h/len(lex)*100:.1f}%)", ha="center", fontsize=9)
    ax.set_ylim(0, max(var_counts.values)*1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lexicon_variant_counts.png"), dpi=300)
    plt.close()

    # Calculate Levenshtein and Normalized Levenshtein distance for all variant pairs
    pair_records = []
    for idx, row in lex.iterrows():
        headword = str(row['Common']).strip()
        for col in var_cols:
            val = row[col]
            if pd.notnull(val):
                variant = str(val).strip()
                if variant.lower() != headword.lower():
                    ld = levenshtein_distance(variant, headword)
                    norm_ld = ld / max(len(variant), len(headword)) if max(len(variant), len(headword)) > 0 else 0
                    pair_records.append({
                        "headword": headword,
                        "variant": variant,
                        "variant_col": col,
                        "levenshtein_distance": ld,
                        "norm_levenshtein_distance": norm_ld,
                        "len_headword": len(headword),
                        "len_variant": len(variant)
                    })
                    
    pairs_df = pd.DataFrame(pair_records)
    pairs_df.to_csv(os.path.join(TABLE_DIR, "lexicon_distance_pairs.csv"), index=False)
    
    # Distance summary stats
    distance_summary = {
        "total_pairs": len(pairs_df),
        "levenshtein_mean": float(pairs_df['levenshtein_distance'].mean()),
        "levenshtein_std": float(pairs_df['levenshtein_distance'].std()),
        "levenshtein_median": float(pairs_df['levenshtein_distance'].median()),
        "levenshtein_min": int(pairs_df['levenshtein_distance'].min()),
        "levenshtein_max": int(pairs_df['levenshtein_distance'].max()),
        "norm_levenshtein_mean": float(pairs_df['norm_levenshtein_distance'].mean()),
        "norm_levenshtein_std": float(pairs_df['norm_levenshtein_distance'].std()),
        "norm_levenshtein_median": float(pairs_df['norm_levenshtein_distance'].median()),
        "distance_frequencies": pairs_df['levenshtein_distance'].value_counts().sort_index().to_dict()
    }
    
    with open(os.path.join(TABLE_DIR, "lexicon_distance_stats.json"), "w", encoding="utf-8") as f:
        json.dump(distance_summary, f, indent=2)

    # Plot edit distance distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw Levenshtein
    dist_counts = pairs_df['levenshtein_distance'].value_counts().sort_index()
    axes[0].bar(dist_counts.index, dist_counts.values, color='#313695', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel("Levenshtein Edit Distance (Character Edits)", fontweight="bold")
    axes[0].set_ylabel("Count of Variant-Common Pairs", fontweight="bold")
    axes[0].set_title(f"Levenshtein Distance Distribution (Mean={distance_summary['levenshtein_mean']:.2f})", fontweight="bold")
    for x_val, count in dist_counts.items():
        if count > 50:
            axes[0].text(x_val, count + 120, f"{count:,}\n({count/len(pairs_df)*100:.1f}%)", ha="center", fontsize=8)
    axes[0].set_ylim(0, max(dist_counts.values)*1.15)
    
    # Normalized Levenshtein
    sns.histplot(pairs_df['norm_levenshtein_distance'], bins=30, kde=True, ax=axes[1], color='#d73027')
    axes[1].axvline(distance_summary['norm_levenshtein_mean'], color='black', linestyle='--', label=f"Mean: {distance_summary['norm_levenshtein_mean']:.3f}")
    axes[1].axvline(distance_summary['norm_levenshtein_median'], color='blue', linestyle=':', label=f"Median: {distance_summary['norm_levenshtein_median']:.3f}")
    axes[1].set_xlabel("Normalized Edit Distance (Edits / Max Word Length)", fontweight="bold")
    axes[1].set_ylabel("Frequency", fontweight="bold")
    axes[1].set_title("Normalized Levenshtein Distance Distribution", fontweight="bold")
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "variant_edit_distances.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Review Coverage & Lexicon Density across Sentiments
    # -------------------------------------------------------------
    logger.info("Computing lexicon coverage and density in reviews...")
    # Map all canonical words and their variants
    lex_word_to_common = {}
    for idx, row in lex.iterrows():
        c = str(row['Common']).strip().lower()
        if c:
            lex_word_to_common[c] = c
            for col in var_cols:
                v = row[col]
                if pd.notnull(v):
                    v_str = str(v).strip().lower()
                    if v_str:
                        lex_word_to_common[v_str] = c

    lex_tokens_set = set(lex_word_to_common.keys())
    
    df_clean['lexicon_matches'] = df_clean['tokens'].apply(lambda toks: sum(1 for t in toks if t in lex_tokens_set))
    df_clean['lexicon_density'] = df_clean.apply(lambda row: (row['lexicon_matches'] / row['word_count']) if row['word_count'] > 0 else 0, axis=1)

    coverage_by_sentiment = df_clean.groupby('sentiment').agg(
        Total_Reviews=('review_id', 'count'),
        Reviews_With_Match=('lexicon_matches', lambda x: (x > 0).sum()),
        Coverage_Pct=('lexicon_matches', lambda x: (x > 0).mean() * 100),
        Mean_Matches=('lexicon_matches', 'mean'),
        Median_Matches=('lexicon_matches', 'median'),
        Mean_Density=('lexicon_density', 'mean'),
        Median_Density=('lexicon_density', 'median')
    ).loc[class_order]
    
    coverage_by_sentiment.round(3).to_csv(os.path.join(TABLE_DIR, "lexicon_coverage_by_sentiment.csv"))

    # Plot Lexicon Density Distribution across classes
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_clean, x='sentiment', y='lexicon_density', order=class_order, ax=ax, palette='Spectral', showfliers=False)
    ax.set_xlabel("Sentiment Class", fontweight="bold")
    ax.set_ylabel("Lexicon Word Density (Matches / Total Tokens)", fontweight="bold")
    ax.set_xticklabels([c.title() for c in class_order])
    ax.set_title("Lexicon Matching Density across Sentiment Classes in RUWV-NSR", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lexicon_density_per_class.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Top Spelling Variations in the Dataset
    # -------------------------------------------------------------
    logger.info("Identifying top frequent spelling variations...")
    from collections import Counter
    token_counter = Counter([t for tokens in df_clean['tokens'] for t in tokens])
    
    variant_review_matches = []
    for idx, row in lex.iterrows():
        c = str(row['Common']).strip().lower()
        variants = [str(row[col]).strip().lower() for col in var_cols if pd.notnull(row[col])]
        total_freq = sum(token_counter.get(w, 0) for w in [c] + variants)
        if total_freq > 0:
            variant_review_matches.append({
                "Common": c,
                "Variants": ", ".join([v for v in variants if v != c]),
                "Num_Variants": len([v for v in variants if v != c]),
                "Total_Occurrences": total_freq,
                "Common_Occurrences": token_counter.get(c, 0),
                "Variant_Occurrences": sum(token_counter.get(v, 0) for v in variants if v != c)
            })
            
    top_var_df = pd.DataFrame(variant_review_matches).sort_values(by="Total_Occurrences", ascending=False).head(20)
    top_var_df.to_csv(os.path.join(TABLE_DIR, "top_20_frequent_words_with_variants.csv"), index=False)
    
    # Save master EDA summary
    eda_summary = {
        "total_cleaned_reviews": len(df_clean),
        "mean_word_count": float(df_clean['word_count'].mean()),
        "median_word_count": float(df_clean['word_count'].median()),
        "p95_word_count": float(p95),
        "mean_ttr": float(df_clean['ttr'].mean()),
        "lexicon_headwords": len(lex),
        "lexicon_total_unique_words": len(lex_tokens_set),
        "reviews_with_lexicon_match_pct": float((df_clean['lexicon_matches'] > 0).mean() * 100),
        "overall_lexicon_density_mean": float(df_clean['lexicon_density'].mean()),
        "distance_stats": distance_summary
    }
    
    with open(os.path.join(TABLE_DIR, "eda_summary_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    logger.info("EDA complete! Figures saved to results/figures/ and tables to results/tables/")
    return eda_summary


if __name__ == "__main__":
    run_eda()
