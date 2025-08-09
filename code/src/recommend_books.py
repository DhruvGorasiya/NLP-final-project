import os
import pandas as pd
import pickle

DATA_PATH = "data"
MODELS_PATH = "models"

BOOKS_FILE = os.path.join(DATA_PATH, "books_with_keywords.csv")
SIM_MATRIX_FILE = os.path.join(MODELS_PATH, "cosine_similarity.pkl")

def load_resources():
    books_df = pd.read_csv(BOOKS_FILE)
    with open(SIM_MATRIX_FILE, "rb") as f:
        sim_matrix = pickle.load(f)
    return books_df, sim_matrix

def recommend_books(title, books_df, sim_matrix, top_n=5):
    title_lower = title.lower()
    matches = books_df[books_df["Book-Title"].str.lower() == title_lower]
    if matches.empty:
        print(f"[WARN] No match for '{title}'")
        return pd.DataFrame()
    idx = matches.index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in scores_sorted[1:top_n+1]]
    return books_df.iloc[top_indices][["Book-Title", "Book-Author", "Publisher"]]

if __name__ == "__main__":
    books_df, sim_matrix = load_resources()
    print(recommend_books("harry potter and the chamber of secrets", books_df, sim_matrix, 5))
