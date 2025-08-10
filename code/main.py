# ==============================
#  main.py — Run full pipeline
# ==============================

import os
from src.preprocessing import clean_books, load_datasets, save_cleaned
from src.keyword_extraction import run_keyword_extraction
from src.vectorization_similarity import run_vectorization_similarity
from src.recommend_books import recommend_books
from src.visualize_test import plot_similarity_heatmap, plot_keyword_importance, load_data

DATA_PATH = "data/"
MODELS_PATH = "models/"

if __name__ == "__main__":
    # 1️⃣ Data Preprocessing
    print("\n[STEP 1] Loading and cleaning dataset...")
    books, users, ratings = load_datasets()
    books_clean = clean_books(books)
    save_cleaned(books_clean)

    # 2️⃣ Keyword Extraction
    print("\n[STEP 2] Extracting keywords with KeyBERT...")
    run_keyword_extraction()

    # 3️⃣ TF-IDF Vectorization & Similarity
    print("\n[STEP 3] Building TF-IDF model and similarity matrix...")
    run_vectorization_similarity()

    # 4️⃣ Example Recommendation
    print("\n[STEP 4] Example Recommendation")
    books_df, sim_matrix = load_data()
    title = "PLEADING GUILTY"
    recs = recommend_books(title, books_df, sim_matrix, top_n=5)
    print(recs)

    # 5️⃣ Visualization Examples
    print("\n[STEP 5] Generating visualizations...")
    plot_similarity_heatmap(books_df, sim_matrix, sample_size=8)
    plot_keyword_importance(title, books_df)
