# ==========================
#  02_keyword_extraction.py
# ==========================

import pandas as pd
from keybert import KeyBERT

# --------------------------
# Configuration
# --------------------------
INPUT_FILE = "data/cleaned_books.csv"
OUTPUT_FILE = "data/books_with_keywords.csv"

BERT_MODEL = "all-MiniLM-L6-v2"   # small, fast SentenceTransformer
NUM_KEYWORDS = 8                  # number of keywords to extract
DIVERSITY = 0.6                    # diversity for Maximal Marginal Relevance

# --------------------------
# Keyword Extraction
# --------------------------
def extract_keywords_from_text(text, kw_model, num_keywords, diversity):
    """
    Extract top-N keywords from a given text using KeyBERT.
    Returns a list of keyword strings.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=num_keywords,
            use_maxsum=True,
            diversity=diversity
        )
        # Return only keyword strings (ignore scores)
        return [kw for kw, _ in keywords]
    except Exception as e:
        print(f"[WARN] Failed to extract keywords: {e}")
        return []

# --------------------------
# Main Pipeline
# --------------------------
if __name__ == "__main__":
    # Load cleaned dataset
    df = pd.read_csv(INPUT_FILE)
    
    # Initialize KeyBERT model
    print(f"[INFO] Loading BERT model: {BERT_MODEL}")
    kw_model = KeyBERT(model=BERT_MODEL)
    
    # Combine relevant text fields into a description
    if "Book-Title" in df.columns and "Book-Author" in df.columns:
        df["full_text"] = df["Book-Title"].astype(str) + " " + df["Book-Author"].astype(str)
    else:
        df["full_text"] = df["Book-Title"].astype(str)
    
    # Extract keywords for each book
    print(f"[INFO] Extracting {NUM_KEYWORDS} keywords per book...")
    df["keywords"] = df["full_text"].apply(
        lambda x: extract_keywords_from_text(x, kw_model, NUM_KEYWORDS, DIVERSITY)
    )
    
    # Save results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved file with keywords to {OUTPUT_FILE}")
