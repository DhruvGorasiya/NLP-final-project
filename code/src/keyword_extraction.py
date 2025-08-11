import os
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from src.experiment_tracking import ExperimentTracker, ExperimentConfig, EXPERIMENT_CONFIGS

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
BATCH_SIZE = 32  # Batch size for processing
MAX_TEXT_LENGTH = 1024  # Maximum text length to process

# --------------------------
# Helpers
# --------------------------
def get_device():
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def preprocess_text(text):
    """Clean and truncate text for keyword extraction."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = str(text).strip()
    # Truncate to max length while preserving word boundaries
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH].rsplit(' ', 1)[0]
    return text

def extract_keywords_from_text(text, kw_model, num_keywords, diversity):
    """Extract keywords from a single text using KeyBERT."""
    text = preprocess_text(text)
    if not text:
        return []
    
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=num_keywords,
            use_maxsum=True,
            diversity=diversity,
            min_df=1,  # Include terms that appear at least once
            threshold=0.3  # Minimum similarity threshold
        )
        return [kw for kw, _ in keywords]
    except Exception as e:
        print(f"[WARN] Keyword extraction failed for text: {text[:50]}... Error: {e}")
        return []

# --------------------------
# Main Function
# --------------------------
def process_batch(texts, kw_model, num_keywords, diversity):
    """Process a batch of texts for keyword extraction."""
    return [extract_keywords_from_text(text, kw_model, num_keywords, diversity) 
            for text in texts]

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
    print("[INFO] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

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

    # Get best available device
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Load model
    print(f"[INFO] Loading BERT model: {bert_model_name}")
    bert_model = SentenceTransformer(bert_model_name, device=device)
    kw_model = KeyBERT(model=bert_model)

    # Process in batches
    keywords_list = []
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_texts = df["full_text"].iloc[start_idx:end_idx].tolist()
        
        batch_keywords = process_batch(batch_texts, kw_model, num_keywords, diversity)
        keywords_list.extend(batch_keywords)
        
        # Free up memory
        if device == "cuda":
            torch.cuda.empty_cache()
    
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
