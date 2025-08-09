import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

DATA_PATH = "data"
MODELS_PATH = "models"

BOOKS_FILE = os.path.join(DATA_PATH, "books_with_keywords.csv")
SIM_MATRIX_FILE = os.path.join(MODELS_PATH, "cosine_similarity.pkl")

def load_data():
    books_df = pd.read_csv(BOOKS_FILE)
    books_df["keywords"] = books_df["keywords"].apply(literal_eval)
    with open(SIM_MATRIX_FILE, "rb") as f:
        sim_matrix = pickle.load(f)
    return books_df, sim_matrix

def plot_similarity_heatmap(books_df, sim_matrix, sample_size=10):
    sample_books = books_df.sample(sample_size, random_state=42)
    indices = sample_books.index.tolist()
    sim_subset = sim_matrix[indices][:, indices]
    plt.figure(figsize=(12, 8))
    sns.heatmap(sim_subset,
                xticklabels=sample_books["Book-Title"],
                yticklabels=sample_books["Book-Title"],
                cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Book Similarity Heatmap (Sample)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_keyword_importance(book_title, books_df):
    row = books_df[books_df["Book-Title"].str.lower() == book_title.lower()]
    if row.empty:
        print(f"[WARN] Book '{book_title}' not found.")
        return
    keywords = row.iloc[0]["keywords"]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[1]*len(keywords), y=keywords, palette="viridis")
    plt.title(f"Keywords for '{book_title}'")
    plt.xlabel("Importance (Not Scaled)")
    plt.ylabel("Keyword")
    plt.show()

if __name__ == "__main__":
    books_df, sim_matrix = load_data()
    plot_similarity_heatmap(books_df, sim_matrix, sample_size=8)
    plot_keyword_importance("harry potter and the chamber of secrets", books_df)
