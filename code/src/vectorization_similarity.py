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
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["keywords"] = df["keywords"].apply(literal_eval)
    print("[INFO] Building TF-IDF matrix...")
    tfidf_matrix, vectorizer = build_tfidf_matrix(df["keywords"])
    with open(TFIDF_MODEL_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Saved TF-IDF vectorizer to {TFIDF_MODEL_FILE}")
    sim_matrix = compute_cosine_similarity(tfidf_matrix)
    with open(SIM_MATRIX_FILE, "wb") as f:
        pickle.dump(sim_matrix, f)
    print(f"[INFO] Saved cosine similarity matrix to {SIM_MATRIX_FILE}")

if __name__ == "__main__":
    run_vectorization_similarity()
