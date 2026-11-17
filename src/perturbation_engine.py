"""
Factorial Perturbation Engine Module
Project: Robustness of Roman Urdu Sentiment Classification to Orthographic Spelling Variation
"""

import os
import re
import json
import random
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLEAN_LEXICON_PATH = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed\cleaned_lexicon.csv"
DATA_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\data_processed"
TABLE_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\tables"


def levenshtein_distance(s1, s2):
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


class PerturbationEngine:
    def __init__(self, lexicon_path=CLEAN_LEXICON_PATH):
        logger.info(f"Initializing PerturbationEngine with lexicon: {lexicon_path}")
        self.lex_df = pd.read_csv(lexicon_path, encoding="utf-8")
        self.word_to_variants = {}
        self._build_lexicon_map()
        
    def _build_lexicon_map(self):
        var_cols = ["var-1", "var-2", "var-3", "var-4", "var-5"]
        for idx, row in self.lex_df.iterrows():
            common = str(row['Common']).strip().lower()
            if not common or common == "nan":
                continue
                
            variants = []
            for col in var_cols:
                v = row[col]
                if pd.notnull(v):
                    v_str = str(v).strip().lower()
                    if v_str and v_str != "nan":
                        variants.append(v_str)
                        
            # Ensure unique variants list
            unique_vars = list(dict.fromkeys(variants))
            
            # Map canonical common word to its variants
            entry = {
                "common": common,
                "all_forms": list(dict.fromkeys([common] + unique_vars)),
                "variants": unique_vars
            }
            
            # Index under common and all variant forms
            for form in entry["all_forms"]:
                if form not in self.word_to_variants:
                    self.word_to_variants[form] = entry
                    
        logger.info(f"Indexed {len(self.word_to_variants)} surface word forms pointing to lexicon entries.")

    def select_variant(self, word, distance_strategy="random", rng=None):
        """
        Select a substitute variant for `word` according to distance strategy:
        - 'low': minimum Levenshtein distance from word
        - 'high': maximum Levenshtein distance from word
        - 'random': uniform random choice from available non-identical variants
        """
        w_lower = word.lower()
        if w_lower not in self.word_to_variants:
            return word, 0
            
        entry = self.word_to_variants[w_lower]
        # Candidate replacements: all forms in entry that are not identical to w_lower
        candidates = [v for v in entry["all_forms"] if v != w_lower]
        if not candidates:
            return word, 0
            
        if distance_strategy == "random":
            chosen = rng.choice(candidates) if rng else random.choice(candidates)
            dist = levenshtein_distance(chosen, w_lower)
            return self._match_casing(word, chosen), dist
            
        # Calculate distances
        cand_dists = [(c, levenshtein_distance(c, w_lower)) for c in candidates]
        
        if distance_strategy == "low":
            # Sort ascending by distance, pick among minimums
            min_d = min(d for c, d in cand_dists)
            min_cands = [c for c, d in cand_dists if d == min_d]
            chosen = rng.choice(min_cands) if rng else random.choice(min_cands)
            return self._match_casing(word, chosen), min_d
        elif distance_strategy == "high":
            # Sort descending by distance, pick among maximums
            max_d = max(d for c, d in cand_dists)
            max_cands = [c for c, d in cand_dists if d == max_d]
            chosen = rng.choice(max_cands) if rng else random.choice(max_cands)
            return self._match_casing(word, chosen), max_d
        else:
            chosen = rng.choice(candidates) if rng else random.choice(candidates)
            dist = levenshtein_distance(chosen, w_lower)
            return self._match_casing(word, chosen), dist

    @staticmethod
    def _match_casing(original, substitute):
        if original.isupper():
            return substitute.upper()
        if original.istitle():
            return substitute.capitalize()
        return substitute

    def perturb_text(self, text, rate=0.5, distance_strategy="random", seed=None):
        """
        Perturb text by replacing lexicon words with orthographic variants.
        """
        rng = random.Random(seed) if seed is not None else random
        tokens = re.findall(r'\b[a-zA-Z]+\b|[^\w\s]|\s+', text)
        
        # Identify word token indices that are in lexicon
        eligible_indices = []
        for i, tok in enumerate(tokens):
            if re.match(r'^[a-zA-Z]+$', tok) and tok.lower() in self.word_to_variants:
                eligible_indices.append(i)
                
        total_eligible = len(eligible_indices)
        if total_eligible == 0 or rate <= 0:
            return text, {
                "eligible_words": 0,
                "perturbed_words": 0,
                "actual_rate": 0.0,
                "mean_edit_distance": 0.0
            }
            
        num_to_perturb = int(round(total_eligible * rate))
        num_to_perturb = max(1, num_to_perturb) if rate > 0 and total_eligible > 0 else 0
        num_to_perturb = min(num_to_perturb, total_eligible)
        
        indices_to_perturb = set(rng.sample(eligible_indices, num_to_perturb))
        
        perturbed_tokens = list(tokens)
        total_dist = 0
        actual_perturbed = 0
        
        for idx in indices_to_perturb:
            orig_word = tokens[idx]
            sub, dist = self.select_variant(orig_word, distance_strategy=distance_strategy, rng=rng)
            if sub.lower() != orig_word.lower():
                perturbed_tokens[idx] = sub
                total_dist += dist
                actual_perturbed += 1
                
        actual_rate = (actual_perturbed / total_eligible) if total_eligible > 0 else 0.0
        mean_dist = (total_dist / actual_perturbed) if actual_perturbed > 0 else 0.0
        
        result_text = "".join(perturbed_tokens)
        return result_text, {
            "eligible_words": total_eligible,
            "perturbed_words": actual_perturbed,
            "actual_rate": actual_rate,
            "mean_edit_distance": mean_dist
        }

    def generate_perturbed_dataset(self, df, rate, distance_strategy, condition_name, seed=42):
        logger.info(f"Generating condition '{condition_name}' (Rate={rate:.2f}, Strategy='{distance_strategy}', N={len(df)})...")
        results = []
        stats_list = []
        
        for idx, row in df.iterrows():
            row_seed = seed + idx
            perturbed_text, stats = self.perturb_text(
                str(row['review']),
                rate=rate,
                distance_strategy=distance_strategy,
                seed=row_seed
            )
            stats_list.append(stats)
            new_row = row.copy()
            new_row['review_original'] = row['review']
            new_row['review'] = perturbed_text
            new_row['condition'] = condition_name
            new_row['target_rate'] = rate
            new_row['distance_strategy'] = distance_strategy
            new_row['perturbed_words'] = stats['perturbed_words']
            new_row['actual_rate'] = stats['actual_rate']
            new_row['mean_edit_distance'] = stats['mean_edit_distance']
            results.append(new_row)
            
        out_df = pd.DataFrame(results)
        
        mean_actual_rate = np.mean([s['actual_rate'] for s in stats_list])
        mean_dist = np.mean([s['mean_edit_distance'] for s in stats_list if s['perturbed_words'] > 0])
        total_perturbed = sum(s['perturbed_words'] for s in stats_list)
        
        logger.info(f"Condition '{condition_name}' complete: Mean Rate={mean_actual_rate*100:.1f}%, Mean Edit Dist={mean_dist:.2f}, Total Words Perturbed={total_perturbed:,}")
        
        condition_summary = {
            "condition": condition_name,
            "target_rate": rate,
            "distance_strategy": distance_strategy,
            "total_reviews": len(out_df),
            "mean_actual_rate": float(mean_actual_rate),
            "mean_edit_distance": float(mean_dist) if not np.isnan(mean_dist) else 0.0,
            "total_perturbed_tokens": int(total_perturbed)
        }
        return out_df, condition_summary


def run_perturbation_pipeline():
    engine = PerturbationEngine()
    test_path = os.path.join(DATA_DIR, "test.csv")
    test_df = pd.read_csv(test_path, encoding="utf-8")
    
    # Standard Canonical Conditions as defined in Proposal Section 6:
    # 1. Clean: rate=0.0
    # 2. Mild: rate=0.20, low distance
    # 3. Moderate: rate=0.50, random distance
    # 4. Severe: rate=0.85, high distance
    
    conditions = [
        {"name": "clean", "rate": 0.0, "strategy": "none"},
        {"name": "mild", "rate": 0.20, "strategy": "low"},
        {"name": "moderate", "rate": 0.50, "strategy": "random"},
        {"name": "severe", "rate": 0.85, "strategy": "high"},
    ]
    
    # Factorial Grid Conditions (3 rates x 3 distances = 9 conditions)
    factorial_rates = [0.25, 0.50, 0.90]
    factorial_strategies = ["low", "random", "high"]
    
    summaries = []
    
    # Generate canonical test sets
    for cond in conditions:
        out_df, summary = engine.generate_perturbed_dataset(
            test_df,
            rate=cond["rate"],
            distance_strategy=cond["strategy"],
            condition_name=cond["name"]
        )
        file_path = os.path.join(DATA_DIR, f"test_{cond['name']}.csv")
        out_df.to_csv(file_path, index=False, encoding="utf-8")
        summaries.append(summary)
        
    # Generate 3x3 Factorial Grid
    logger.info("Generating 3x3 Factorial Grid conditions...")
    for r in factorial_rates:
        for s in factorial_strategies:
            grid_name = f"grid_rate{int(r*100)}_{s}"
            out_df, summary = engine.generate_perturbed_dataset(
                test_df,
                rate=r,
                distance_strategy=s,
                condition_name=grid_name
            )
            file_path = os.path.join(DATA_DIR, f"test_{grid_name}.csv")
            out_df.to_csv(file_path, index=False, encoding="utf-8")
            summaries.append(summary)
            
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(TABLE_DIR, "perturbation_conditions_summary.csv"), index=False)
    
    with open(os.path.join(TABLE_DIR, "perturbation_conditions_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
        
    logger.info("All perturbation datasets successfully generated and saved to data_processed/")
    return summaries


if __name__ == "__main__":
    run_perturbation_pipeline()
