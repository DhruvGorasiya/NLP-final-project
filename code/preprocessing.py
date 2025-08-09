# ==========================
#  01_preprocessing.py
# ==========================

import pandas as pd
import numpy as np
import re
import string

# --------------------------
# Configuration
# --------------------------
DATA_PATH = "data/"
BOOKS_FILE = DATA_PATH + "Books.csv"
USERS_FILE = DATA_PATH + "Users.csv"
RATINGS_FILE = DATA_PATH + "Ratings.csv"
OUTPUT_FILE = DATA_PATH + "cleaned_books.csv"

# --------------------------
# Load Data
# --------------------------
def load_datasets():
    """Read the Kaggle Book Recommendation dataset."""
    books = pd.read_csv(BOOKS_FILE, encoding='latin-1')
    users = pd.read_csv(USERS_FILE, encoding='latin-1')
    ratings = pd.read_csv(RATINGS_FILE, encoding='latin-1')
    return books, users, ratings

# --------------------------
# Cleaning Helpers
# --------------------------
def strip_html(text):
    """Remove HTML tags."""
    return re.sub(r"<.*?>", "", str(text))

def remove_urls(text):
    """Remove web URLs."""
    return re.sub(r"https?://\S+|www\.\S+", "", str(text))

def clean_text(text):
    """Basic NLP cleaning: lowercase, remove punctuation, extra spaces."""
    text = str(text).lower()
    text = remove_urls(strip_html(text))
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------
# Book Cleaning
# --------------------------
def clean_books(books_df):
    """Clean and filter the Books dataframe."""
    df = books_df.copy()
    
    # Drop invalid publication years
    df = df[(df["Year-Of-Publication"] >= 1800) & (df["Year-Of-Publication"] <= 2025)]
    
    # Keep only English books if language info exists
    if "Language" in df.columns:
        df = df[df["Language"] == "eng"]
    
    # Clean text fields
    for col in ["Book-Title", "Book-Author", "Publisher"]:
        df[col] = df[col].apply(clean_text)
    
    return df

# --------------------------
# Save Cleaned Data
# --------------------------
def save_cleaned(df, file_path=OUTPUT_FILE):
    df.to_csv(file_path, index=False)
    print(f"[INFO] Saved cleaned data to {file_path}")

# --------------------------
# Main Pipeline
# --------------------------
if __name__ == "__main__":
    books, users, ratings = load_datasets()
    books_clean = clean_books(books)
    save_cleaned(books_clean)
