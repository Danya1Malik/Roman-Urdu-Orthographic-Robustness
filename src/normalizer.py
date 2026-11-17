"""
Orthographic and Preprocessing Normalization Module
Implements four normalization conditions for research ablation:
  (A) Raw / No-norm
  (B) Generic NLP Preprocessing (elongation collapsing, URL/emoji removal, punctuation/whitespace cleanup)
  (C) Lexicon-Based Orthographic Normalization (mapping variants to Common canonical forms)
  (D) Combined Generic Preprocessing + Lexicon Normalization
"""

import os
import re
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLEAN_LEXICON_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_lexicon.csv"


class RomanUrduNormalizer:
    def __init__(self, lexicon_path=CLEAN_LEXICON_PATH):
        logger.info(f"Initializing RomanUrduNormalizer with lexicon: {lexicon_path}")
        self.lex_df = pd.read_csv(lexicon_path, encoding="utf-8")
        self.variant_to_common = {}
        self._build_normalization_dict()
        
    def _build_normalization_dict(self):
        var_cols = ["var-1", "var-2", "var-3", "var-4", "var-5"]
        for idx, row in self.lex_df.iterrows():
            common = str(row['Common']).strip().lower()
            if not common or common == "nan":
                continue
            for col in var_cols:
                v = row[col]
                if pd.notnull(v):
                    v_str = str(v).strip().lower()
                    if v_str and v_str != "nan" and v_str != common:
                        self.variant_to_common[v_str] = common
        logger.info(f"Built normalization dictionary mapping {len(self.variant_to_common):,} non-canonical variants to Common forms.")

    @staticmethod
    def generic_preprocess(text):
        """
        Generic informal text preprocessing:
        - URL and HTML stripping
        - Mention and hashtag symbol cleanup
        - Character elongation collapse (e.g. 'bohooooot' -> 'bohoot')
        - Whitespace normalization
        """
        if not isinstance(text, str):
            return ""
            
        t = text
        # Remove URLs
        t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
        # Remove HTML tags
        t = re.sub(r'<.*?>', ' ', t)
        # Remove mentions
        t = re.sub(r'@\w+', ' ', t)
        # Remove hashtag characters but keep word
        t = re.sub(r'#(\w+)', r'\1', t)
        # Collapse character elongations (3 or more consecutive identical characters -> 2)
        t = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', t)
        # Remove non-ASCII symbols and special characters except basic punctuation
        t = re.sub(r'[^\w\s\.\?!,]', ' ', t)
        # Collapse multiple spaces
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def lexicon_normalize(self, text):
        """
        Research-specific orthographic normalization:
        Replace known spelling variants with canonical Common forms while preserving casing.
        """
        if not isinstance(text, str):
            return ""
            
        def replace_token(match):
            word = match.group(0)
            w_lower = word.lower()
            if w_lower in self.variant_to_common:
                canonical = self.variant_to_common[w_lower]
                if word.isupper():
                    return canonical.upper()
                if word.istitle():
                    return canonical.capitalize()
                return canonical
            return word

        return re.sub(r'\b[a-zA-Z]+\b', replace_token, text)

    def normalize(self, text, condition="raw"):
        """
        Apply normalization condition:
        - 'raw': no normalization
        - 'generic': generic preprocessing only
        - 'lexicon': lexicon normalization only
        - 'combined': generic preprocessing + lexicon normalization
        """
        cond = condition.lower().strip()
        if cond in ["raw", "none", "a"]:
            return text
        elif cond in ["generic", "b"]:
            return self.generic_preprocess(text)
        elif cond in ["lexicon", "c"]:
            return self.lexicon_normalize(text)
        elif cond in ["combined", "d"]:
            cleaned = self.generic_preprocess(text)
            return self.lexicon_normalize(cleaned)
        else:
            raise ValueError(f"Unknown normalization condition: {condition}")

    def normalize_dataframe(self, df, text_col="review", condition="combined"):
        df_out = df.copy()
        df_out[text_col] = df_out[text_col].apply(lambda t: self.normalize(str(t), condition=condition))
        return df_out


if __name__ == "__main__":
    normalizer = RomanUrduNormalizer()
    sample = "allah naahi usko bachannaa tha bohhhooooot ghandi choice hai www.site.com @user"
    print("Original: ", sample)
    print("Generic:  ", normalizer.normalize(sample, "generic"))
    print("Lexicon:  ", normalizer.normalize(sample, "lexicon"))
    print("Combined: ", normalizer.normalize(sample, "combined"))
