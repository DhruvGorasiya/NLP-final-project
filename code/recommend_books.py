# ==============================
#  04_recommend_books.py
# ==============================

import pandas as pd
import pickle
import numpy as np

# --------------------------
# Config
# --------------------------
BOOKS_FILE = "data/books_with_keywords.csv"
TFIDF_MODEL_FILE = "models/tfidf_vectorizer.pkl"
SIM_MATRIX_FILE = "models/cosine_similarity.pkl"

# --------------------------
# Load Data & Models
# --------------------------
def load_resources():
    """Load books dataset, TF-IDF vectorizer, and cosine similarity matrix."""
    books_df = pd.read_csv(BOOKS_FILE)
    
    with open(TFIDF_MODEL_FILE, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    
    with open(SIM_MATRIX_FILE, "rb") as f:
        sim_matrix = pickle.load(f)
    
    return books_df, tfidf_vectorizer, sim_matrix

# --------------------------
# Recommendation Function
# --------------------------
def recommend_books(title, books_df, sim_matrix, top_n=5):
    """
    Given a book title, return top-N similar books.
    """
    # Normalize title search
    title_lower = title.lower()
    
    if "Book-Title" not in books_df.columns:
        raise ValueError("Books dataframe must have 'Book-Title' column")
    
    matches = books_df[books_df["Book-Title"].str.lower() == title_lower]
    
    if matches.empty:
        print(f"[WARN] No match found for '{title}'.")
        return []
    
    # Take first matching index
    idx = matches.index[0]
    
    # Get similarity scores for that book
    scores = list(enumerate(sim_matrix[idx]))
    
    # Sort by score (descending), skip self (idx)
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Get top-N similar book indices
    top_indices = [i for i, _ in scores_sorted[1:top_n+1]]
    
    # Return corresponding book rows
    return books_df.iloc[top_indices][["Book-Title", "Book-Author", "Publisher"]]

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    books_df, tfidf_vectorizer, sim_matrix = load_resources()
    
    book_name = "harry potter and the chamber of secrets"  # Example
    recommendations = recommend_books(book_name, books_df, sim_matrix, top_n=5)
    
    print(f"\nTop 5 books similar to '{book_name}':\n")
    print(recommendations)
