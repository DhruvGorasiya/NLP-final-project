# ==============================
#  03_vectorization_and_similarity.py
# ==============================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --------------------------
# Config
# --------------------------
INPUT_FILE = "data/books_with_keywords.csv"
TFIDF_MODEL_FILE = "models/tfidf_vectorizer.pkl"
SIM_MATRIX_FILE = "models/cosine_similarity.pkl"

# --------------------------
# Vectorization Function
# --------------------------
def build_tfidf_matrix(keyword_lists):
    """
    Convert list-of-keywords into TF-IDF matrix.
    """
    # Join keyword lists into space-separated strings
    docs = [" ".join(keywords) for keywords in keyword_lists]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    return tfidf_matrix, vectorizer

# --------------------------
# Similarity Function
# --------------------------
def compute_cosine_similarity(tfidf_matrix):
    """
    Compute cosine similarity between all items.
    """
    return cosine_similarity(tfidf_matrix)

# --------------------------
# Main Pipeline
# --------------------------
if __name__ == "__main__":
    # Load dataset with keywords
    df = pd.read_csv(INPUT_FILE)
    
    # Build TF-IDF matrix
    print("[INFO] Building TF-IDF matrix...")
    tfidf_matrix, vectorizer = build_tfidf_matrix(df["keywords"].apply(eval))
    
    # Save TF-IDF model
    with open(TFIDF_MODEL_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Saved TF-IDF vectorizer to {TFIDF_MODEL_FILE}")
    
    # Compute cosine similarity
    print("[INFO] Computing cosine similarity...")
    sim_matrix = compute_cosine_similarity(tfidf_matrix)
    
    # Save similarity matrix
    with open(SIM_MATRIX_FILE, "wb") as f:
        pickle.dump(sim_matrix, f)
    print(f"[INFO] Saved cosine similarity matrix to {SIM_MATRIX_FILE}")
