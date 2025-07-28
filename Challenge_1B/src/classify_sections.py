# classify_sections.py
# Uses the trained DistilBERT model to identify section titles in PDF lines

import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import os

MODEL_PATH = "models/distilbert_section_classifier"
INPUT_PATH = "data/parsed_lines.json"
OUTPUT_PATH = "data/section_candidates.json"

LABEL_ID2NAME = {0: "O", 1: "B-TITLE", 2: "I-TITLE"}

# Load model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

def is_title(label_ids):
    return any(l in [1, 2] for l in label_ids)

def classify_line(text):
    inputs = tokenizer(text.split(), return_tensors="pt", is_split_into_words=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs).logits
    preds = torch.argmax(outputs, dim=-1).squeeze().tolist()
    return preds

def main():
    with open(INPUT_PATH, "r") as f:
        lines = json.load(f)

    section_candidates = []

    for line in lines:
        tokens = line["text"].split()
        if len(tokens) == 0:
            continue

        label_ids = classify_line(line["text"])
        if is_title(label_ids):
            section_candidates.append({
                "document": line["source_pdf"],
                "section_title": line["text"],
                "page_number": line["page_number"],
                "confidence": sum(1 for l in label_ids if l in [1, 2]) / len(label_ids)
            })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(section_candidates, f, indent=2)
    print(f"Classified {len(section_candidates)} section titles → saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
