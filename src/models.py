"""
Model Architecture Definitions and Wrappers
Supports:
  1. TF-IDF + Logistic Regression Baseline
  2. mBERT (bert-base-multilingual-cased) Full Fine-Tuning
  3. XLM-RoBERTa (xlm-roberta-base) Full Fine-Tuning
  4. XLM-RoBERTa LoRA (PEFT Adapter Fine-Tuning)
"""

import os
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

LABEL_TO_ID = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

LABEL_TO_3CLASS = {
    "very negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very positive": "positive"
}


def build_tfidf_baseline(max_features=25000, ngram_range=(1, 3), C=1.0, random_state=42):
    """
    Constructs a competitive TF-IDF + Logistic Regression pipeline.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w+\b'
        )),
        ('clf', LogisticRegression(
            C=C,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline


def get_transformer_tokenizer(model_name="xlm-roberta-base"):
    return AutoTokenizer.from_pretrained(model_name)


def get_transformer_model(
    model_name="xlm-roberta-base",
    num_labels=5,
    use_lora=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1
):
    """
    Loads Transformer model for sequence classification.
    Optionally applies PEFT LoRA adapters.
    """
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    if use_lora:
        # Determine target modules based on model family
        if "xlm-roberta" in model_name.lower():
            target_modules = ["query", "value"]
        elif "bert" in model_name.lower():
            target_modules = ["query", "value"]
        else:
            target_modules = ["query", "value"]
            
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        print("LoRA Model Parameter Summary:")
        model.print_trainable_parameters()
        
    return model


if __name__ == "__main__":
    print("Testing TF-IDF Baseline initialization...")
    tfidf = build_tfidf_baseline()
    print("TF-IDF pipeline created.")
    
    print("\nTesting LoRA XLM-R Model initialization...")
    xlmr_lora = get_transformer_model("xlm-roberta-base", use_lora=True)
    print("LoRA XLM-R successfully initialized.")
