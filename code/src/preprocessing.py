import os
import pandas as pd
import numpy as np
import re
import string

# --------------------------
# Config
# --------------------------
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

BOOKS_FILE = os.path.join(DATA_PATH, "Books.csv")
USERS_FILE = os.path.join(DATA_PATH, "Users.csv")
RATINGS_FILE = os.path.join(DATA_PATH, "Ratings.csv")
OUTPUT_FILE = os.path.join(DATA_PATH, "cleaned_books.csv")

# --------------------------
# Helpers
# --------------------------
def strip_html(text):
    return re.sub(r"<.*?>", "", str(text))

def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", str(text))

def clean_text(text):
    text = str(text).lower()                                    # Convert to lowercase
    text = remove_urls(strip_html(text))                        # Remove URLs and HTML
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return re.sub(r"\s+", " ", text).strip()                   # Remove extra spaces

# --------------------------
# Main Functions
# --------------------------
def load_datasets():
    books = pd.read_csv(BOOKS_FILE, encoding="latin-1")
    users = pd.read_csv(USERS_FILE, encoding="latin-1")
    ratings = pd.read_csv(RATINGS_FILE, encoding="latin-1")
    return books, users, ratings

def clean_books(books_df):
    df = books_df.copy()
    # Keep original title for display
    df["Original-Title"] = df["Book-Title"]
    # Clean text fields
    for col in ["Book-Title", "Book-Author", "Publisher"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    return df

def save_cleaned(df):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved cleaned books to {OUTPUT_FILE}")

if __name__ == "__main__":
    books, users, ratings = load_datasets()
    books_clean = clean_books(books)
    save_cleaned(books_clean)
