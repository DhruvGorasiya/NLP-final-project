import os
import pandas as pd
import numpy as np
import re
import string
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

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
def load_datasets(max_rows=10000):
    """Load datasets with option to limit number of rows."""
    books = pd.read_csv(BOOKS_FILE, encoding="latin-1")
    users = pd.read_csv(USERS_FILE, encoding="latin-1")
    ratings = pd.read_csv(RATINGS_FILE, encoding="latin-1")
    
    if max_rows:
        print(f"[INFO] Limiting dataset to {max_rows} rows")
        books = books.head(max_rows)
        # Get only the users and ratings related to these books
        valid_isbns = books['ISBN'].tolist()
        ratings = ratings[ratings['ISBN'].isin(valid_isbns)]
        valid_users = ratings['User-ID'].unique()
        users = users[users['User-ID'].isin(valid_users)]
    
    print(f"[INFO] Dataset sizes:")
    print(f"Books: {len(books)}")
    print(f"Users: {len(users)}")
    print(f"Ratings: {len(ratings)}")
    
    return books, users, ratings

def validate_year(year):
    try:
        year = int(year)
        current_year = 2024
        return 1800 <= year <= current_year
    except:
        return False

def validate_age(age):
    try:
        age = float(age)
        return 5 <= age <= 110  # Reasonable age range
    except:
        return False

def detect_language(text):
    try:
        return detect(str(text)) if text else None
    except LangDetectException:
        return None

def clean_books(books_df):
    df = books_df.copy()
    
    # Keep original title for display
    df["Original-Title"] = df["Book-Title"]
    
    # Clean text fields
    for col in ["Book-Title", "Book-Author", "Publisher"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Detect language from title and description if available
    text_for_lang = df["Book-Title"].str.cat(df["Book-Author"], sep=" ", na_rep="")
    if "Description" in df.columns:
        text_for_lang = text_for_lang.str.cat(df["Description"], sep=" ", na_rep="")
    
    df["Language"] = text_for_lang.apply(detect_language)
    
    # Filter English-only books
    df = df[df["Language"] == "en"].copy()
    df = df.drop("Language", axis=1)
    
    # Validate and clean publication year
    if "Year-Of-Publication" in df.columns:
        df["Valid-Year"] = df["Year-Of-Publication"].apply(validate_year)
        df.loc[~df["Valid-Year"], "Year-Of-Publication"] = None
        df = df.drop("Valid-Year", axis=1)
    
    # Extract and validate image URLs
    for size in ["Small", "Medium", "Large"]:
        url_col = f"Image-URL-{size}"
        if url_col in df.columns:
            df[f"Has-{size}-Image"] = df[url_col].notna() & df[url_col].str.startswith(("http://", "https://"))
    
    print(f"[INFO] Filtered to {len(df)} English books")
    return df

def clean_users(users_df):
    df = users_df.copy()
    
    # Clean and validate age
    if "Age" in df.columns:
        df["Valid-Age"] = df["Age"].apply(validate_age)
        df.loc[~df["Valid-Age"], "Age"] = None
        df = df.drop("Valid-Age", axis=1)
    
    # Clean location data
    if "Location" in df.columns:
        df["Location"] = df["Location"].fillna("Unknown")
        df["Location"] = df["Location"].apply(clean_text)
    
    return df

def save_cleaned(df):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved cleaned books to {OUTPUT_FILE}")

def create_interaction_matrix(books_df, users_df, ratings_df):
    # Create user-book interaction matrix
    matrix = pd.pivot_table(
        ratings_df,
        values="Book-Rating",
        index="User-ID",
        columns="ISBN",
        fill_value=0
    )
    
    # Add implicit feedback (1 for rated, 0 for not rated)
    implicit_matrix = (matrix > 0).astype(int)
    
    return matrix, implicit_matrix

if __name__ == "__main__":
    # Load and clean datasets
    books, users, ratings = load_datasets()
    books_clean = clean_books(books)
    users_clean = clean_users(users)
    
    # Create interaction matrices
    explicit_matrix, implicit_matrix = create_interaction_matrix(books_clean, users_clean, ratings)
    
    # Save cleaned data
    books_clean.to_csv(os.path.join(DATA_PATH, "cleaned_books.csv"), index=False)
    users_clean.to_csv(os.path.join(DATA_PATH, "cleaned_users.csv"), index=False)
    
    # Save interaction matrices
    explicit_matrix.to_csv(os.path.join(DATA_PATH, "explicit_interactions.csv"))
    implicit_matrix.to_csv(os.path.join(DATA_PATH, "implicit_interactions.csv"))
    
    print("[INFO] Preprocessing complete. Files saved to data directory.")
