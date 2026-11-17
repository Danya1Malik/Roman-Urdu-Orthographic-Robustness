# Robustness of Roman Urdu Sentiment Classification to Orthographic Spelling Variation

**Comparative Study of Full Fine-Tuning and Parameter-Efficient Adaptation of Multilingual Transformers**  
**Course:** Data Science Tools & Techniques (DS T&T), Fall 2026  
**Institution:** Department of Data Science — Graduate Studies, FAST-NUCES  
**Author:** Danyal Ahmed Malik (26i-8030)  
**Email:** i268030@isb.nu.edu.pk  

---

## 📌 Executive Summary

Roman Urdu serves as the primary vernacular medium for informal digital communication across South Asia, yet it lacks standardized orthography. The resulting spelling variation significantly degrades natural language processing pipelines.

This repository contains an end-to-end, reproducible, leakage-free empirical investigation into the orthographic robustness of multilingual transformer models (**mBERT** and **XLM-RoBERTa**) compared against a classical **TF-IDF + Logistic Regression** baseline, alongside parameter-efficient adaptation via **LoRA (Low-Rank Adaptation)**.

Key contributions:
1. **Independent Data Audit & Leakage-Free Stratification:** Resolved 8 disparate raw label strings into 5 canonical classes and eliminated **4,632 exact duplicate reviews (16.49%)** to construct a strictly leakage-free benchmark split ($N_{\text{train}}=16,420$, $N_{\text{val}}=3,519$, $N_{\text{test}}=3,519$).
2. **Disentangled 3×3 Factorial Grid:** Separated word substitution density (`{25%, 50%, 90%}`) from character edit distance (`Low`, `Random`, `High`).
3. **Subword Fragmentation Profiling:** Quantified subword fertility across **10,869 authentic lexical variant pairs**, demonstrating that XLM-RoBERTa's unigram vocabulary suffers less structural fragmentation than mBERT.
4. **Controlled Experiments:** Executed zero-shot degradation (**Exp A**), robust data augmentation (**Exp B**), lexicon normalization ablation (**Exp C**), and external cross-domain transfer (**Stage 4** on 26,824-review RUECD corpus).

---

## 📂 Project Structure & Folder Organization

The project is structured according to professional scientific software engineering standards:

```text
Project/
├── Dataset/                                # Raw unprocessed source datasets
│   ├── Roman Urdu reviews Dataset with English translation.csv
│   ├── Roman Urdu words and English Translation with spelling variation.csv
│   └── RUECD.csv                           # External e-commerce test corpus
├── data_processed/                         # Sanitized, deduplicated, and split datasets
│   ├── cleaned_reviews.csv                 # Cleaned & deduplicated reviews (23,458 rows)
│   ├── cleaned_lexicon.csv                 # Inverted headword-to-variant lexicon
│   ├── cleaned_ruecd.csv                   # Validated external corpus
│   ├── train.csv / val.csv / test.csv      # Leakage-free stratified splits (70/15/15)
│   ├── train_augmented.csv                 # 50% synthetic variant-augmented train set
│   ├── test_clean.csv                      # Test condition: 0% perturbation
│   ├── test_mild.csv / moderate / severe   # Canonical perturbation conditions
│   └── test_grid_rate{25,50,90}_{low,random,high}.csv  # 3x3 factorial grid test sets
├── notebooks/                              # Interactive, reproducible Jupyter Notebooks
│   ├── 01_data_validation_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_dataset_split_and_perturbation.ipynb
│   ├── 04_model_architectures_and_training.ipynb
│   ├── 05_experiments_and_statistical_evaluation.ipynb
│   └── 06_error_analysis_and_confusion_matrices.ipynb
├── src/                                    # Reusable modular pipeline codebase
│   ├── __init__.py
│   ├── data_validation.py                  # Audit, deduplication, and schema validation
│   ├── dataset_split.py                    # Stratified partitioning & perturbation generator
│   ├── eda.py                              # Morphological profiling & summary statistics
│   ├── models.py                           # Model architectures (TF-IDF, mBERT, XLM-R, LoRA)
│   ├── normalizer.py                       # Lexicon-based and rule-based normalizers
│   ├── perturbation_engine.py              # Authentic variant substitution engine
│   ├── statistical_tests.py                # Bootstrap CIs & McNemar significance tests
│   ├── tokenization_analysis.py            # WordPiece vs SentencePiece fertility metrics
│   ├── train.py                            # Checkpoint training loops with early stopping
│   ├── evaluate.py                         # Evaluation metrics (Accuracy, Macro-F1, Confusion Matrix)
│   └── external_validation.py              # Cross-domain validation on RUECD
├── experiments/                            # Standalone experiment execution scripts
│   ├── experiment_A_zeroshot.py            # Exp A: Zero-shot degradation & 3x3 factorial grid
│   ├── experiment_B_augmented.py           # Exp B: Robust data augmentation
│   ├── experiment_C_normalization.py       # Exp C: Conditions A, B, C, D ablation & density
│   └── run_all_experiments.py              # Master orchestrator for all experiments
├── checkpoints/                            # Saved trained model weights & configs
│   ├── tfidf_baseline.joblib
│   ├── tfidf_augmented.joblib
│   ├── full_mbert/                         # bert-base-multilingual-cased checkpoint
│   ├── full_xlm_roberta/                   # xlm-roberta-base checkpoint
│   ├── lora_xlm_roberta/                   # PEFT LoRA adapter checkpoint (clean train)
│   └── lora_xlm_roberta_augmented/         # PEFT LoRA adapter checkpoint (augmented train)
├── results/                                # Output artifacts, metrics, and visual figures
│   ├── figures/                            # Publication-quality charts (300 DPI PNGs)
│   ├── confusion_matrices/                 # Detailed per-condition error matrices
│   ├── tables/                             # CSV & JSON metrics, significance tests, splits
│   └── logs/                               # Execution logs for auditing
├── report/
│   └── RESEARCH_PROPOSAL_FINAL.md          # Comprehensive academic research proposal
└── README.md                               # Project documentation & pipeline guide
```

---

## 🔄 End-to-End Data Pipeline Architecture

### Pipeline Flowchart

```
[ Raw Mendeley RUWV-NSR ]                  [ Raw Lexicon (5,219 headwords) ]
           │                                                │
           ▼                                                ▼
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│ 1. Data Validation & Audit      │        │ 1. Lexicon Inversion & Mapping  │
│  - Strip nulls & artifacts      │        │  - Lowercase & strip whitespace │
│  - Map 8 raw strings -> 5 labels│        │  - Build variant mapping index  │
│  - Eliminate 4,632 duplicates   │        │  - 10,869 authentic pairs       │
└────────────────┬────────────────┘        └────────────────┬────────────────┘
                 │                                          │
                 ▼                                          │
    [ cleaned_reviews.csv (23,458) ]                        │
                 │                                          │
                 ▼                                          │
┌─────────────────────────────────┐                         │
│ 2. Leakage-Free Stratification  │                         │
│  - Seed 42, 70 / 15 / 15 split  │                         │
│  - Train (16,420)               │                         │
│  - Val   (3,519)                │                         │
│  - Test  (3,519)                │                         │
└────────────────┬────────────────┘                         │
                 │                                          │
                 ├──────────────────────────────────────────┤
                 ▼                                          ▼
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│ 3. Perturbation Engine          │◄───────┤ Authentic Lexicon Lookups       │
│  - Canonical: Mild, Mod, Severe │        │  - Low, Random, High Edit Dist  │
│  - Factorial: 3x3 Grid (9 sets) │        │  - Levenshtein distance metrics │
│  - Augmentation: 50% noisy train│        └─────────────────────────────────┘
└────────────────┬────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 4. Model Training & Parameter Adaptation                 │
│  - Classical: TF-IDF + Logistic Regression Baseline       │
│  - Full Fine-Tuning: mBERT (178M) & XLM-RoBERTa (278M)   │
│  - Parameter-Efficient: XLM-RoBERTa LoRA (1.77M params)  │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 5. Experimental Evaluation & Statistical Testing         │
│  - Exp A: Zero-shot degradation & 3x3 ANOVA              │
│  - Exp B: Robust data augmentation comparison            │
│  - Exp C: Normalization recovery (Cond A, B, C, D)       │
│  - Stage 4: Cross-domain transfer on RUECD corpus        │
│  - Paired Bootstrap CIs (B=1000) & McNemar significance  │
└──────────────────────────────────────────────────────────┘
```

### Detailed Pipeline Stages

#### 1. Ingestion, Validation & Deduplication (`src/data_validation.py`)
* **Raw Ingestion:** Reads Mendeley RUWV-NSR dataset (`28,090` rows) with encoding fallback handling (`latin-1`, `cp1252`, `utf-8`).
* **Label Canonicalization:** Resolves raw annotations into 5 consistent classes:
  * `Positive` (4,548) + `positive` (1,740) $\rightarrow$ `positive` (6,288)
  * `Negative` (3,975) + `negative` (2,060) + `Neative` (typo: 1) $\rightarrow$ `negative` (6,036)
  * `neutral` $\rightarrow$ `neutral` (6,445)
  * `very positive` $\rightarrow$ `very positive` (5,008)
  * `very negative` $\rightarrow$ `very negative` (4,313)
* **Deduplication:** Identifies and eliminates **4,632 exact duplicate reviews (16.49%)**, resolving contradictory annotations. Retains **23,458** unique, pristine reviews.
* **Lexicon Inversion:** Processes 5,219 headwords into an inverted index of **10,869 authentic variant pairs** with calculated Levenshtein edit distances.

#### 2. Stratified Partitioning (`src/dataset_split.py`)
* Performs a strictly leakage-free stratified split using random seed `42`:
  * **Train Set (70%):** 16,420 reviews
  * **Validation Set (15%):** 3,519 reviews
  * **Test Set (15%):** 3,519 reviews
* Retains exact class distribution across all splits (~25.9% Neutral, ~25.2% Positive, ~24.4% Negative, ~13.5% Very Positive, ~11.0% Very Negative).

#### 3. Perturbation Engine & Factorial Design (`src/perturbation_engine.py`)
* Replaces canonical headwords in text exclusively with authentic lexical variants documented in the lexicon.
* **Canonical Severity Conditions:**
  * `Clean`: 0% perturbation.
  * `Mild`: 25% substitution probability, random edit distance.
  * `Moderate`: 50% substitution probability, random edit distance.
  * `Severe`: 90% substitution probability, random edit distance.
* **3×3 Factorial Grid:** Disentangles substitution rate (`25%`, `50%`, `90%`) from edit distance (`Low`, `Random`, `High`) yielding 9 separate test distributions.
* **Data Augmentation:** Injects authentic variants into 50% of the training split to produce `train_augmented.csv` (24,630 rows).

#### 4. Model Training & Adaptation (`src/train.py`, `src/models.py`)
* **TF-IDF + Logistic Regression:** Sublinear TF-IDF with 10,000 character/word n-grams $(1, 3)$, $L_2$ penalty, $C=1.0$.
* **Full Multilingual Fine-Tuning:**
  * `bert-base-multilingual-cased`: 178M parameters, AdamW $\text{lr}=2\times 10^{-5}$, linear warmup, early stopping.
  * `xlm-roberta-base`: 278M parameters, AdamW $\text{lr}=2\times 10^{-5}$, linear warmup, early stopping.
* **Parameter-Efficient LoRA Adaptation:**
  * `xlm-roberta-base` with Low-Rank Adapters targeting attention query and value matrices ($r=16, \alpha=32, \text{dropout}=0.1$).
  * Only **1,770,757 trainable parameters (~0.63%)**, reducing GPU VRAM allocation from ~5.2 GB to ~3.1 GB.

#### 5. Evaluation & Statistical Testing (`src/evaluate.py`, `src/statistical_tests.py`)
* Computes Macro-F1, Accuracy, Precision, Recall, and Confusion Matrices across all conditions.
* Paired non-parametric bootstrap resampling ($B=1,000$) for 95% confidence intervals.
* Exact McNemar's tests to establish statistical significance ($p < 0.05$) between models.

---

## 📊 Empirical Results & Visualizations

### 1. Zero-Shot Degradation under Orthographic Noise (Experiment A)

| Model | Clean Acc (F1) | Mild Acc (F1) | Moderate Acc (F1) | Severe Acc (F1) | Relative Drop |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Reg** | 0.6559 (0.6417) | 0.5891 (0.5707) | 0.5283 (0.5050) | 0.4430 (0.4079) | **-32.46%** |
| **mBERT (Full FT)** | 0.7306 (0.7226) | 0.6777 (0.6677) | 0.6272 (0.6133) | 0.5516 (0.5312) | **-24.50%** |
| **XLM-RoBERTa (Full FT)** | 0.7423 (0.7337) | 0.6974 (0.6865) | 0.6547 (0.6402) | 0.5843 (0.5630) | **-21.28%** |
| **XLM-RoBERTa (LoRA)** | **0.7431 (0.7350)** | **0.7022 (0.6917)** | **0.6635 (0.6508)** | **0.6010 (0.5831)** | **-19.12%** |

![Experiment A Robustness Curves](results/figures/experiment_A_robustness_curves.png)
*Figure 1: Accuracy degradation curves across canonical severity levels. LoRA adapter exhibits the lowest relative performance degradation.*

![Experiment A Relative Drop](results/figures/experiment_A_relative_drop.png)
*Figure 2: Percentage degradation relative to clean baseline. LoRA demonstrates superior regularization compared to full fine-tuning.*

---

### 2. Disentangling Rate vs Edit Distance ($3 \times 3$ Factorial Grid)

![Experiment A Factorial Heatmap](results/figures/experiment_A_factorial_heatmap.png)
*Figure 3: Factorial ANOVA heatmap separating token substitution density (rows) from character edit distance magnitude (columns).*

---

### 3. Subword Tokenization Fragmentation

![Tokenizer Fragmentation](results/figures/tokenizer_fragmentation_comparison.png)
*Figure 4: Token fertility ratio between canonical headwords and authentic variants across WordPiece (mBERT) and SentencePiece/BPE (XLM-RoBERTa).*

---

### 4. Corpus Morphological Distributions

![Class Distribution](results/figures/class_distribution.png)
*Figure 5: Class balance comparison between raw scraped corpus and cleaned unique corpus.*

![Review Length Distribution](results/figures/review_length_distribution.png)
*Figure 6: Review length distribution (token count) stratified across the 5 sentiment classes.*

![Edit Distance Distribution](results/figures/variant_edit_distances.png)
*Figure 7: Distribution of Levenshtein edit distances across authentic Roman Urdu variant pairs.*

---

### 5. Data Augmentation Resilience (Experiment B)

![Experiment B Comparison](results/figures/experiment_B_augmented_comparison.png)
*Figure 8: Performance recovery when training models with 50% synthetic authentic variant augmentation.*

---

### 6. Normalization Recovery & Density Predictability (Experiment C)

![Normalization Recovery](results/figures/experiment_C_normalization_recovery.png)
*Figure 9: Performance across Condition A (Raw Perturbed), Condition B (Generic Preprocessing), Condition C (Lexicon Normalization), and Condition D (Preprocessing + Normalization).*

![Density vs Recovery](results/figures/experiment_C_density_vs_recovery.png)
*Figure 10: Performance recovery as a function of review spelling variant density.*

---

### 7. Confusion Matrices (Clean vs Severe Noise)

| Model | Clean Test Matrix | Severe Noise Matrix |
| :---: | :---: | :---: |
| **XLM-RoBERTa (LoRA)** | ![LoRA Clean](results/confusion_matrices/cm_xlm_roberta_lora_clean.png) | ![LoRA Severe](results/confusion_matrices/cm_xlm_roberta_lora_severe.png) |
| **XLM-RoBERTa (Full)** | ![XLM-R Clean](results/confusion_matrices/cm_xlm_roberta_full_clean.png) | ![XLM-R Severe](results/confusion_matrices/cm_xlm_roberta_full_severe.png) |
| **mBERT (Full)** | ![mBERT Clean](results/confusion_matrices/cm_mbert_full_clean.png) | ![mBERT Severe](results/confusion_matrices/cm_mbert_full_severe.png) |
| **TF-IDF + LR** | ![TF-IDF Clean](results/confusion_matrices/cm_tf_idf_+_lr_clean.png) | ![TF-IDF Severe](results/confusion_matrices/cm_tf_idf_+_lr_severe.png) |

---

## 💻 Interactive Jupyter Notebooks (`notebooks/`)

The repository includes a sequential suite of 6 structured `.ipynb` notebooks:

1. **`01_data_validation_and_cleaning.ipynb`**:
   - Ingests raw reviews and lexicon.
   - Executes label canonicalization (8 $\rightarrow$ 5 classes).
   - Detects and removes 4,632 duplicate reviews.
   - Generates and displays `results/tables/data_validation_report.json` and `results/figures/class_distribution.png`.

2. **`02_exploratory_data_analysis.ipynb`**:
   - Analyzes review length distributions, vocabulary coverage, and type-token ratios.
   - Evaluates Levenshtein edit distance distributions across 10,869 variant pairs.
   - Subword fragmentation analysis comparing WordPiece (mBERT) and SentencePiece (XLM-RoBERTa).

3. **`03_dataset_split_and_perturbation.ipynb`**:
   - Performs stratified 70/15/15 train/val/test splitting.
   - Constructs canonical conditions (`Clean`, `Mild`, `Moderate`, `Severe`).
   - Builds $3 \times 3$ Factorial Grid test sets and augmented training corpus (`train_augmented.csv`).

4. **`04_model_architectures_and_training.ipynb`**:
   - Documents TF-IDF baseline, mBERT (Full), XLM-RoBERTa (Full), and XLM-RoBERTa (LoRA).
   - Reports trainable parameters, parameter efficiency, VRAM requirements, and training loss histories.

5. **`05_experiments_and_statistical_evaluation.ipynb`**:
   - Reproduces and displays results for **Experiment A**, **Experiment B**, **Experiment C**, and **Stage 4 (External Validation on RUECD)**.
   - Displays degradation curves, relative drops, factorial ANOVA heatmaps, and statistical significance tests.

6. **`06_error_analysis_and_confusion_matrices.ipynb`**:
   - Inspects confusion matrices across clean and perturbed test conditions.
   - Analyzes inter-class boundary shifts and misclassification patterns.
   - Reviews the master experiment manifest.

---

## 🚀 Reproduction & Setup Guide

### 1. Environment Installation

Ensure Python 3.10+ is installed with CUDA-compatible PyTorch:

```bash
# Clone repository
cd "c:\Users\Daama\Desktop\DSTT PROJECT"

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets scikit-learn pandas numpy matplotlib seaborn jupyterlab
```

### 2. Running via Jupyter Notebooks

Launch JupyterLab to interact with notebooks:

```bash
jupyter lab notebooks/
```

### 3. Running via Python Command Line

Run individual pipeline components or execute the full experiment suite:

```bash
# Step 1: Validate and clean datasets
python -m src.data_validation

# Step 2: Run Exploratory Data Analysis & Tokenizer Fragmentation
python -m src.eda
python -m src.tokenization_analysis

# Step 3: Partition dataset and generate perturbation conditions
python -m src.dataset_split

# Step 4: Run all experiments (Exp A, Exp B, Exp C, External Validation)
python -m experiments.run_all_experiments
```

---

## 📚 Academic References & Citations

1. **Nazir et al. (IEEE Access, 2025):** Leveraging Multilingual Transformer for Multiclass Sentiment Analysis in Code-Mixed Data of Low-Resource Languages. *IEEE Access*, vol. 13, pp. 7538–7554. [DOI: 10.1109/ACCESS.2025.3527299](https://doi.org/10.1109/ACCESS.2025.3527299)
2. **Khan et al. (arXiv, 2026):** ROMEVA: Geometry-Preserving Vocabulary Expansion for Roman Urdu Language Models. *arXiv:2606.22478*.
3. **Hu et al. (ICLR, 2022):** LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations*.
4. **Conneau et al. (ACL, 2020):** Unsupervised Cross-Lingual Representation Learning at Scale (XLM-RoBERTa). *Proceedings of ACL 2020*, pp. 8440–8451.
5. **Devlin et al. (NAACL, 2019):** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL 2019*, pp. 4171–4186.
6. **Ahmed (Mendeley Data, 2024):** Roman Urdu Word Variations and Normalized Sentiment Review Dataset (RUWV-NSR), V5. [DOI: 10.17632/v5jfhsvtmd.5](https://doi.org/10.17632/v5jfhsvtmd.5)
7. **Chandio et al. (CMES, 2022):** Sentiment Analysis of Roman Urdu on E-Commerce Reviews Using Machine Learning. *Computer Modeling in Engineering & Sciences*, 131(3), pp. 1263–1287. [DOI: 10.32604/cmes.2022.018382](https://doi.org/10.32604/cmes.2022.018382)
8. **Moradi & Samwald (EMNLP, 2021):** Evaluating the Robustness of Neural Language Models to Input Perturbations. *Proceedings of EMNLP 2021*, pp. 1558–1570.
9. **Loshchilov & Hutter (ICLR, 2019):** Decoupled Weight Decay Regularization (AdamW). *International Conference on Learning Representations*.
10. **Wolf et al. (EMNLP, 2020):** Transformers: State-of-the-Art Natural Language Processing. *Proceedings of EMNLP 2020 System Demonstrations*, pp. 38–45.
11. **Kudo & Richardson (EMNLP, 2018):** SentencePiece: A Simple and Language Independent Subword Alternative for Neural Text Processing. *Proceedings of EMNLP 2018 System Demonstrations*, pp. 66–71.
12. **Paszke et al. (NeurIPS, 2019):** PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, vol. 32, pp. 8024–8035.
13. **Dettmers et al. (NeurIPS, 2023):** QLoRA: Efficient Finetuning of Quantized LLMs. *Advances in Neural Information Processing Systems*, vol. 36, pp. 10088–10115.
