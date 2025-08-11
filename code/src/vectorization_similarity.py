import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from ast import literal_eval
from sentence_transformers import SentenceTransformer
import torch

DATA_PATH = "data"
MODELS_PATH = "models"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "books_with_keywords.csv")
TFIDF_MODEL_FILE = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")
BERT_MODEL_FILE = os.path.join(MODELS_PATH, "bert_embeddings.pkl")
SIM_MATRIX_FILE = os.path.join(MODELS_PATH, "cosine_similarity.pkl")

# BERT model configuration
BERT_MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller, faster model good for semantic similarity

def build_tfidf_matrix(keyword_lists):
    """Build TF-IDF matrix from keyword lists."""
    docs = [" ".join(keywords) for keywords in keyword_lists]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    return tfidf_matrix, vectorizer

def build_bert_embeddings(texts, device=None):
    """Generate BERT embeddings for texts."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"[INFO] Using device: {device}")
    model = SentenceTransformer(BERT_MODEL_NAME, device=device)
    
    # Process in batches to manage memory
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device
        )
        embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)

def compute_cosine_similarity(matrix_a, matrix_b=None):
    """Compute cosine similarity between matrices."""
    if matrix_b is None:
        matrix_b = matrix_a
    return cosine_similarity(matrix_a, matrix_b)

def blend_similarities(sim_matrix_1, sim_matrix_2, weight=0.5):
    """Blend two similarity matrices with given weight."""
    return weight * sim_matrix_1 + (1 - weight) * sim_matrix_2

def run_vectorization_similarity():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")
    
    # Load and prepare data
    df = pd.read_csv(INPUT_FILE)
    df["keywords"] = df["keywords"].apply(literal_eval)
    
    # Build TF-IDF representations
    print("[INFO] Building TF-IDF matrix...")
    tfidf_matrix, vectorizer = build_tfidf_matrix(df["keywords"])
    with open(TFIDF_MODEL_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Saved TF-IDF vectorizer to {TFIDF_MODEL_FILE}")
    
    # Generate BERT embeddings
    print("[INFO] Generating BERT embeddings...")
    texts = [" ".join(keywords) for keywords in df["keywords"]]
    bert_embeddings = build_bert_embeddings(texts)
    with open(BERT_MODEL_FILE, "wb") as f:
        pickle.dump(bert_embeddings, f)
    print(f"[INFO] Saved BERT embeddings to {BERT_MODEL_FILE}")
    
    # Compute similarity matrices
    print("[INFO] Computing TF-IDF similarities...")
    tfidf_sim_matrix = compute_cosine_similarity(tfidf_matrix)
    
    print("[INFO] Computing BERT similarities...")
    bert_sim_matrix = compute_cosine_similarity(bert_embeddings)
    
    # Blend similarities
    print("[INFO] Blending similarity matrices...")
    final_sim_matrix = blend_similarities(tfidf_sim_matrix, bert_sim_matrix)
    
    # Save final similarity matrix
    with open(SIM_MATRIX_FILE, "wb") as f:
        pickle.dump(final_sim_matrix, f)
    print(f"[INFO] Saved blended similarity matrix to {SIM_MATRIX_FILE}")

if __name__ == "__main__":
    run_vectorization_similarity()
