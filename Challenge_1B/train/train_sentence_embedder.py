# train_sentence_embedder.py
# Trains a lightweight ML model on sentence embeddings to classify section headers.

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Load and prepare data (ensure 'text' and 'label' columns exist)
df = pd.read_csv("./data/section_lines.csv")  # Contains text,label columns

# Load embedding model (offline mode assumed)
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB

# Compute embeddings
embeddings = embedder.encode(df["text"].tolist(), show_progress_bar=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, df["label"], test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and embedder
joblib.dump(clf, "models/sentence_embedder/classifier.joblib")
embedder.save("models/sentence_embedder/encoder")
print("Sentence embedder and classifier saved.")
