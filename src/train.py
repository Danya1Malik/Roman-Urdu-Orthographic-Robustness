"""
Model Training Module
Supports training:
  1. TF-IDF + Logistic Regression
  2. mBERT (bert-base-multilingual-cased) Full Fine-Tuning
  3. XLM-RoBERTa (xlm-roberta-base) Full Fine-Tuning
  4. XLM-RoBERTa LoRA (PEFT Adapter Fine-Tuning)
"""

import os
import time
import json
import logging
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np

from src.models import (
    build_tfidf_baseline,
    get_transformer_model,
    get_transformer_tokenizer,
    LABEL_TO_ID,
    ID_TO_LABEL
)
from src.evaluate import compute_classification_metrics

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHECKPOINT_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\checkpoints"
LOG_DIR = r"C:\Users\Daama\Desktop\DSTT PROJECT\results\logs"


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = list(texts)
        self.labels = [LABEL_TO_ID[l] for l in labels] if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_tfidf(train_df, val_df, save_name="tfidf_baseline.joblib"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, save_name)
    
    logger.info("Training TF-IDF + Logistic Regression baseline...")
    start_time = time.time()
    model = build_tfidf_baseline()
    model.fit(train_df["review"].astype(str), train_df["sentiment"])
    train_time = time.time() - start_time
    
    logger.info(f"TF-IDF training completed in {train_time:.2f} seconds. Saving to {save_path}")
    joblib.dump(model, save_path)
    
    val_preds = model.predict(val_df["review"].astype(str))
    val_metrics = compute_classification_metrics(val_df["sentiment"], val_preds)
    logger.info(f"TF-IDF Validation: Accuracy={val_metrics['accuracy']:.4f}, Macro-F1={val_metrics['macro_f1']:.4f}")
    
    return model, val_metrics


def train_transformer(
    model_name="xlm-roberta-base",
    use_lora=False,
    train_df=None,
    val_df=None,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    max_length=64,
    save_name=None,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    tag = ("lora_" if use_lora else "full_") + model_name.split("/")[-1].replace("-", "_")
    if save_name is None:
        save_name = tag
        
    save_dir = os.path.join(CHECKPOINT_DIR, save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"--- Starting Training: {tag} on {device} (Epochs={epochs}, BatchSize={batch_size}, LR={lr}) ---")
    
    tokenizer = get_transformer_tokenizer(model_name)
    model = get_transformer_model(model_name=model_name, num_labels=5, use_lora=use_lora)
    model.to(device)
    
    train_dataset = TextClassificationDataset(train_df["review"], train_df["sentiment"], tokenizer, max_length=max_length)
    val_dataset = TextClassificationDataset(val_df["review"], val_df["sentiment"], tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    best_val_f1 = 0.0
    best_metrics = {}
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_t = time.time()
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        epoch_time = time.time() - start_t
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend([ID_TO_LABEL[p] for p in preds])
                val_targets.extend([ID_TO_LABEL[l.item()] for l in batch["labels"]])
                
        val_metrics = compute_classification_metrics(val_targets, val_preds)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        
        logger.info(f"Epoch {epoch}/{epochs} [{epoch_time:.1f}s]: Train Loss={avg_train_loss:.4f} | Val Acc={val_acc:.4f} | Val Macro-F1={val_f1:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = val_metrics
            logger.info(f"--> Best model improved to Macro-F1={best_val_f1:.4f}. Saving checkpoint...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
    # Save training history
    with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    logger.info(f"Training complete for {tag}. Best Val Macro-F1: {best_val_f1:.4f}")
    return model, tokenizer, best_metrics
