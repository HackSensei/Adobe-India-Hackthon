# summarize_section.py
# Generates extractive summaries using TF-IDF ranking

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize


def summarize(text, max_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text  # No need to summarize

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(X, X)
    scores = sim_matrix.sum(axis=1)
    ranked = np.argsort(scores)[::-1]

    selected = sorted(ranked[:max_sentences])
    summary = " ".join([sentences[i] for i in selected])
    return summary


# Example
if __name__ == "__main__":
    para = """
    This paper proposes a new transformer architecture. Prior work focused on self-attention but ignored efficiency. We show a 20% reduction in FLOPs with competitive accuracy. Experiments confirm gains across 3 datasets. We conclude with recommendations for real-world deployment.
    """
    print(summarize(para))
