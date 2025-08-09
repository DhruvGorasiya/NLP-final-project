import os
import pandas as pd
from keybert import KeyBERT

DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "cleaned_books.csv")
OUTPUT_FILE = os.path.join(DATA_PATH, "books_with_keywords.csv")

BERT_MODEL = "all-MiniLM-L6-v2"
NUM_KEYWORDS = 8
DIVERSITY = 0.6

def extract_keywords_from_text(text, kw_model, num_keywords, diversity):
    if not isinstance(text, str) or not text.strip():
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
        return [kw for kw, _ in keywords]
    except Exception as e:
        print(f"[WARN] Keyword extraction failed: {e}")
        return []

def run_keyword_extraction():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Loading BERT model: {BERT_MODEL}")
    kw_model = KeyBERT(model=BERT_MODEL)
    if "Book-Author" in df.columns:
        df["full_text"] = df["Book-Title"].astype(str) + " " + df["Book-Author"].astype(str)
    else:
        df["full_text"] = df["Book-Title"].astype(str)
    print(f"[INFO] Extracting {NUM_KEYWORDS} keywords...")
    df["keywords"] = df["full_text"].apply(
        lambda x: extract_keywords_from_text(x, kw_model, NUM_KEYWORDS, DIVERSITY)
    )
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved file with keywords to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_keyword_extraction()
