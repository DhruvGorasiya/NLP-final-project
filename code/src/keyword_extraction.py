import os
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from .experiment_tracking import ExperimentTracker, ExperimentConfig, EXPERIMENT_CONFIGS

# --------------------------
# Config
# --------------------------
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "cleaned_books.csv")
OUTPUT_FILE = os.path.join(DATA_PATH, "books_with_keywords.csv")

BERT_MODEL = "all-MiniLM-L6-v2"
NUM_KEYWORDS = 8
DIVERSITY = 0.6
MAX_ROWS = 500  # For quick testing — set None for full dataset

# --------------------------
# Helpers
# --------------------------
def extract_keywords_from_text(text, kw_model, num_keywords, diversity):
    """Extract keywords from a single text using KeyBERT."""
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

# --------------------------
# Main Function
# --------------------------
def run_keyword_extraction(config: ExperimentConfig = None):
    """Run keyword extraction with optional experiment config."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    # Initialize experiment tracking if config provided
    tracker = ExperimentTracker() if config else None
    if tracker:
        tracker.start_experiment(config)
        # Use config parameters instead of globals
        bert_model_name = config.bert_model
        num_keywords = config.num_keywords
        diversity = config.diversity
    else:
        # Use default parameters
        bert_model_name = BERT_MODEL
        num_keywords = NUM_KEYWORDS
        diversity = DIVERSITY

    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    # For testing, limit rows
    if MAX_ROWS:
        df = df.head(MAX_ROWS)
        print(f"[INFO] Limiting to first {MAX_ROWS} rows for testing.")

    # Use description if available, else title + author
    if "Description" in df.columns:
        df["full_text"] = df["Description"].fillna(
            df["Book-Title"] + " " + df["Book-Author"].astype(str)
        )
    else:
        df["full_text"] = df["Book-Title"] + " " + df["Book-Author"].astype(str)

    # Remove duplicates for efficiency
    df = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)
    print(f"[INFO] Unique text rows after deduplication: {len(df)}")

    # Detect device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load model
    bert_model = SentenceTransformer(bert_model_name, device=device)
    kw_model = KeyBERT(model=bert_model)

    # Extract keywords with progress bar
    keywords_list = []
    for text in tqdm(df["full_text"], desc="Extracting keywords"):
        keywords_list.append(extract_keywords_from_text(text, kw_model, num_keywords, diversity))

    df["keywords"] = keywords_list

    # Generate output filename based on config
    if config:
        output_file = os.path.join(DATA_PATH, f"books_with_keywords_{config.name}.csv")
    else:
        output_file = OUTPUT_FILE

    # Save results
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved file with keywords to {output_file}")

    return df, output_file

# --------------------------
# Script Execution
# --------------------------
if __name__ == "__main__":
    run_keyword_extraction()
