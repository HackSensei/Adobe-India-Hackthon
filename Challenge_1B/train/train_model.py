import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import joblib

# Load CSV
df = pd.read_csv("training_data.csv")
df = df.dropna(subset=["text", "label"])

# Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])
joblib.dump(le, "label_encoder.joblib")

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label_encoded"].tolist(), test_size=0.2, random_state=42
)

# Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Prepare HuggingFace Datasets
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})
val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels
})

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# Training args
training_args = TrainingArguments(
    output_dir="./models/pdf_heading_classifier",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save
trainer.save_model("models/pdf_heading_classifier")
tokenizer.save_pretrained("models/pdf_heading_classifier")

print("Model trained and saved successfully.")