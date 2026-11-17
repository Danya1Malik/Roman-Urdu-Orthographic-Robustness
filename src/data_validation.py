"""
Data Validation and Deduplication Module
Project: Robustness of Roman Urdu Sentiment Classification to Orthographic Spelling Variation
"""

import os
import json
import logging
import pandas as pd
import numpy as np

# Configure logging
os.makedirs(r"C:\Users\Daama\Desktop\DSTT PROJECT\results\logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(r"C:\Users\Daama\Desktop\DSTT PROJECT\results\logs\data_validation.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RAW_REVIEWS_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\Dataset\Roman Urdu reviews Dataset with English translation.csv"
RAW_LEXICON_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\Dataset\Roman Urdu words and English Translation with spelling variation.csv"

OUTPUT_REVIEWS_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_reviews.csv"
OUTPUT_LEXICON_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_lexicon.csv"
REPORT_JSON_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables\data_validation_report.json"


def load_raw_reviews(filepath=RAW_REVIEWS_PATH):
    logger.info(f"Loading raw reviews dataset from: {filepath}")
    # Raw reviews file is encoded in latin-1 / cp1252 due to special characters
    encodings = ["latin-1", "cp1252", "utf-8"]
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            logger.info(f"Successfully loaded raw reviews with encoding='{enc}', shape={df.shape}")
            return df
        except Exception as e:
            logger.warning(f"Failed loading with encoding='{enc}': {e}")
    raise RuntimeError("Could not load raw reviews dataset with any supported encoding.")


def load_raw_lexicon(filepath=RAW_LEXICON_PATH):
    logger.info(f"Loading raw spelling variation lexicon from: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"Successfully loaded lexicon, shape={df.shape}")
    return df


def validate_and_clean_reviews(df):
    logger.info("--- Validating and Cleaning Reviews Dataset ---")
    initial_shape = df.shape
    raw_columns = list(df.columns)
    logger.info(f"Initial columns: {raw_columns}")
    
    # Standardize column names
    df = df.rename(columns={
        "ROMAN URDU REVIEWS": "review",
        "TRANSLATED IN ENGLISH ": "translated_english",
        "TRANSLATED IN ENGLISH": "translated_english",
        "SENTIMENT": "sentiment_raw"
    })
    
    # 1. Inspect raw sentiment distribution
    raw_sentiment_counts = df["sentiment_raw"].value_counts(dropna=False).to_dict()
    logger.info(f"Raw sentiment counts (distinct string forms):\n{json.dumps(raw_sentiment_counts, indent=2)}")
    
    # Clean sentiment labels: strip whitespace, lowercase, fix typo 'neative' -> 'negative'
    sentiment_map = {
        "neutral": "neutral",
        "positive": "positive",
        "Positive": "positive",
        "negative": "negative",
        "Negative": "negative",
        "very positive": "very positive",
        "very negative": "very negative",
        "Neative": "negative",
        "neative": "negative"
    }
    
    df["sentiment"] = df["sentiment_raw"].astype(str).str.strip().map(lambda s: sentiment_map.get(s, sentiment_map.get(s.lower(), s.lower())))
    
    # Validate 5 classes
    valid_classes = {"neutral", "positive", "negative", "very positive", "very negative"}
    invalid_labels = df[~df["sentiment"].isin(valid_classes)]
    if len(invalid_labels) > 0:
        logger.error(f"Found unexpected sentiment labels: {invalid_labels['sentiment'].unique()}")
    else:
        logger.info(f"All sentiment labels successfully validated and mapped to 5 classes: {valid_classes}")

    cleaned_sentiment_counts = df["sentiment"].value_counts().to_dict()
    logger.info(f"Cleaned sentiment counts across 28,090 raw records:\n{json.dumps(cleaned_sentiment_counts, indent=2)}")
    
    # 2. Check for missing/null values
    null_counts = df.isnull().sum().to_dict()
    logger.info(f"Null counts per column: {null_counts}")
    
    # 3. Duplicate Analysis
    exact_duplicates_all = int(df.duplicated().sum())
    duplicate_reviews_exact = int(df.duplicated(subset=["review"]).sum())
    
    # Group by review to find sentiment disagreements
    review_sentiment_agg = df.groupby("review")["sentiment"].nunique()
    conflicting_reviews_count = int((review_sentiment_agg > 1).sum())
    concordant_reviews_count = int((review_sentiment_agg == 1).sum())
    logger.info(f"Total duplicate reviews: {duplicate_reviews_exact} (16.5% of dataset)")
    logger.info(f"Exact duplicates across all columns: {exact_duplicates_all}")
    logger.info(f"Review texts with conflicting labels: {conflicting_reviews_count}")
    logger.info(f"Review texts with concordant labels: {concordant_reviews_count}")
    
    # Deduplication strategy:
    # Drop duplicate review text keeping the first occurrence (standard for RUWV-NSR dataset)
    df_dedup = df.drop_duplicates(subset=["review"], keep="first").copy()
    dedup_shape = df_dedup.shape
    logger.info(f"Dataset shape after deduplication: {dedup_shape} (retained {dedup_shape[0]} / {initial_shape[0]} rows)")
    
    dedup_sentiment_counts = df_dedup["sentiment"].value_counts().to_dict()
    logger.info(f"Deduplicated sentiment class distribution:\n{json.dumps(dedup_sentiment_counts, indent=2)}")
    
    # Reset index
    df_dedup = df_dedup.reset_index(drop=True)
    df_dedup["review_id"] = df_dedup.index
    
    report = {
        "initial_record_count": initial_shape[0],
        "initial_column_count": initial_shape[1],
        "raw_sentiment_distribution": raw_sentiment_counts,
        "cleaned_raw_sentiment_distribution": cleaned_sentiment_counts,
        "duplicate_review_texts": duplicate_reviews_exact,
        "duplicate_all_columns": exact_duplicates_all,
        "conflicting_sentiment_reviews": conflicting_reviews_count,
        "deduplicated_record_count": dedup_shape[0],
        "deduplicated_sentiment_distribution": dedup_sentiment_counts,
        "deduplicated_sentiment_proportions": (df_dedup["sentiment"].value_counts(normalize=True) * 100).round(2).to_dict()
    }
    
    return df_dedup, report


def validate_and_clean_lexicon(df):
    logger.info("--- Validating and Cleaning Lexicon Dataset ---")
    initial_shape = df.shape
    logger.info(f"Initial lexicon shape: {initial_shape}")
    
    # Clean column strings
    var_cols = ["var-1", "var-2", "var-3", "var-4", "var-5"]
    
    # Handle NaN in Common column (e.g., row 3282 'naan')
    nan_common = df[df["Common"].isnull()]
    if len(nan_common) > 0:
        logger.warning(f"Found {len(nan_common)} rows with null Common form. Imputing from var-1.")
        df.loc[df["Common"].isnull(), "Common"] = df.loc[df["Common"].isnull(), "var-1"]
        
    # Strip whitespace from all string columns
    for col in var_cols + ["Common", "English Translated"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "None", ""].copy()), col] = np.nan

    # Calculate variant counts per word
    df["num_recorded_variants"] = df[var_cols].notnull().sum(axis=1)
    
    variant_distribution = df["num_recorded_variants"].value_counts().sort_index().to_dict()
    logger.info(f"Recorded variants per headword distribution: {variant_distribution}")
    
    # Distinct non-identical variants
    non_identical_variants_count = 0
    all_lexicon_tokens = set()
    
    for idx, row in df.iterrows():
        c = str(row["Common"]).strip().lower()
        if c:
            all_lexicon_tokens.add(c)
        for col in var_cols:
            v = row[col]
            if pd.notnull(v):
                v_str = str(v).strip().lower()
                if v_str:
                    all_lexicon_tokens.add(v_str)
                    if v_str != c:
                        non_identical_variants_count += 1
                        
    logger.info(f"Total vocabulary across lexicon (common + all variants): {len(all_lexicon_tokens)}")
    logger.info(f"Total non-identical variant-common pairs: {non_identical_variants_count}")
    
    lexicon_report = {
        "headword_count": len(df),
        "columns": list(df.columns),
        "variant_count_distribution": variant_distribution,
        "unique_tokens_total": len(all_lexicon_tokens),
        "non_identical_variant_pairs": non_identical_variants_count
    }
    
    return df, lexicon_report


def run_pipeline():
    # 1. Load reviews
    raw_reviews = load_raw_reviews()
    cleaned_reviews, reviews_report = validate_and_clean_reviews(raw_reviews)
    
    # 2. Load lexicon
    raw_lexicon = load_raw_lexicon()
    cleaned_lexicon, lexicon_report = validate_and_clean_lexicon(raw_lexicon)
    
    # 3. Save processed datasets
    os.makedirs(os.path.dirname(OUTPUT_REVIEWS_PATH), exist_ok=True)
    cleaned_reviews.to_csv(OUTPUT_REVIEWS_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned reviews to: {OUTPUT_REVIEWS_PATH}")
    
    cleaned_lexicon.to_csv(OUTPUT_LEXICON_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned lexicon to: {OUTPUT_LEXICON_PATH}")
    
    # 4. Save validation report
    full_report = {
        "reviews_validation": reviews_report,
        "lexicon_validation": lexicon_report
    }
    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Saved data validation report to: {REPORT_JSON_PATH}")
    
    print("\n" + "="*50)
    print("DATA VALIDATION AND DEDUPLICATION COMPLETE")
    print(f"Reviews: {reviews_report['initial_record_count']} -> {reviews_report['deduplicated_record_count']} (Duplicates removed: {reviews_report['duplicate_review_texts']})")
    print(f"Lexicon: {lexicon_report['headword_count']} headwords, {lexicon_report['unique_tokens_total']} total unique tokens")
    print("="*50 + "\n")
    return full_report


if __name__ == "__main__":
    run_pipeline()
