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
MAX_ROWS = 5000  # For quick testing — set None for full dataset

# --------------------------
# Helpers
# --------------------------
def extract_keywords_from_text(text, kw_model, num_keywords, diversity):
    """Extract keywords from a single text using KeyBERT with fallback strategies."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        # First try with use_maxsum=False (more reliable for short texts)
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=num_keywords,
            use_maxsum=False,  # Fixed: Changed from True to False
            diversity=diversity
        )
        
        if keywords:  # If we got keywords, return them
            return [kw for kw, _ in keywords]
        
        # If no keywords with use_maxsum=False, try with different parameters
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),  # Try single words only
            stop_words=None,  # No stop words
            top_n=num_keywords,
            use_maxsum=False,
            diversity=diversity
        )
        
        if keywords:
            return [kw for kw, _ in keywords]
        
        # If still no keywords, try with even more lenient settings
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_n=min(num_keywords, 5),  # Fewer keywords
            use_maxsum=False,
            diversity=0.3  # Lower diversity
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

    # Create enhanced text using metadata for better keyword extraction
    def enhance_text_with_metadata(row):
        """Enhance the text with additional metadata to improve keyword extraction."""
        enhanced_text = f"{row['Book-Title']} by {row['Book-Author']}"
        
        # Add publisher if available
        publisher = row.get("Publisher")
        if publisher and isinstance(publisher, str) and publisher.strip():
            enhanced_text += f" published by {publisher}"
        
        # Add year if available
        year = row.get("Year-Of-Publication")
        if year and isinstance(year, (int, str)) and str(year).isdigit():
            enhanced_text += f" in {year}"
        
        return enhanced_text
    
    df["full_text"] = df.apply(enhance_text_with_metadata, axis=1)

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
    successful_extractions = 0
    
    for text in tqdm(df["full_text"], desc="Extracting keywords"):
        keywords = extract_keywords_from_text(text, kw_model, num_keywords, diversity)
        keywords_list.append(keywords)
        if keywords:
            successful_extractions += 1

    df["keywords"] = keywords_list

    # Print statistics
    print(f"[INFO] Successful keyword extractions: {successful_extractions}/{len(df)} ({successful_extractions/len(df)*100:.1f}%)")
    print(f"[INFO] Books with no keywords: {len(df) - successful_extractions}")

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
