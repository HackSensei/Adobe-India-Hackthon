# train_section_classifier.py
# Fine-tunes DistilBERT to classify PDF lines as section titles or not

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

# ------------------- CONFIG -------------------
LABELS = ["O", "B-TITLE", "I-TITLE"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
MODEL_SAVE_PATH = "models/distilbert_section_classifier"

# ------------------- DATASET -------------------
df = pd.read_csv("data/training_data.csv")  # Has columns: tokens, ner_tags (list of int)

# Convert to HuggingFace Dataset
raw_dataset = Dataset.from_pandas(df)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(example):
    tokens = example["tokens"]
    labels = example["ner_tags"]
    tokenized_inputs = tokenizer(tokens, truncation=True, padding="max_length", max_length=128, is_split_into_words=True)

    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(labels[word_idx])
        else:
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

# Tokenize
encoded_dataset = raw_dataset.map(tokenize_and_align_labels)

# Split
train_test = encoded_dataset.train_test_split(test_size=0.2)

# ------------------- MODEL -------------------
model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# ------------------- TRAINER -------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=10,
    save_total_limit=2,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer
)

trainer.train()

# Save
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("Section classifier trained and saved.")
