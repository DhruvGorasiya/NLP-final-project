import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from ast import literal_eval

DATA_PATH = "data"
MODELS_PATH = "models"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "books_with_keywords.csv")
TFIDF_MODEL_FILE = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")
SIM_MATRIX_FILE = os.path.join(MODELS_PATH, "cosine_similarity.pkl")

def build_tfidf_matrix(keyword_lists):
    docs = [" ".join(keywords) for keywords in keyword_lists]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    return tfidf_matrix, vectorizer

def compute_cosine_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)

def run_vectorization_similarity():
    """Run vectorization and similarity computation with improved statistics."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE)
    df["keywords"] = df["keywords"].apply(literal_eval)
    
    print(f"[INFO] Loaded {len(df)} books with keywords")
    
    # Check keyword statistics
    books_with_keywords = df[df["keywords"].apply(len) > 0]
    print(f"[INFO] Books with keywords: {len(books_with_keywords)}")
    print(f"[INFO] Books without keywords: {len(df) - len(books_with_keywords)}")
    
    print("[INFO] Building TF-IDF matrix...")
    tfidf_matrix, vectorizer = build_tfidf_matrix(df["keywords"])
    
    print(f"[INFO] TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"[INFO] TF-IDF matrix density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.6f}")
    
    # Save TF-IDF vectorizer
    with open(TFIDF_MODEL_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Saved TF-IDF vectorizer to {TFIDF_MODEL_FILE}")
    
    # Compute similarity
    sim_matrix = compute_cosine_similarity(tfidf_matrix)
    print(f"[INFO] Similarity matrix shape: {sim_matrix.shape}")
    print(f"[INFO] Max similarity: {sim_matrix.max():.4f}")
    print(f"[INFO] Min similarity: {sim_matrix.min():.4f}")
    print(f"[INFO] Mean similarity: {sim_matrix.mean():.4f}")
    
    # Save similarity matrix
    with open(SIM_MATRIX_FILE, "wb") as f:
        pickle.dump(sim_matrix, f)
    print(f"[INFO] Saved cosine similarity matrix to {SIM_MATRIX_FILE}")
    
    return df, sim_matrix

if __name__ == "__main__":
    run_vectorization_similarity()
