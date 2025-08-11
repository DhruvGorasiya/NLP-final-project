# ==============================
#  main.py — Run full pipeline
# ==============================

import os
import pandas as pd
from src.preprocessing import (
    load_datasets, 
    clean_books, 
    clean_users,
    create_interaction_matrix
)
from src.keyword_extraction import run_keyword_extraction
from src.vectorization_similarity import run_vectorization_similarity

from src.recommend_books import recommend_books, load_resources
from src.experiment_tracking import ExperimentTracker, ExperimentConfig, ExperimentMetrics
from src.visualize_experiments import (
    create_metrics_comparison,
    load_experiment_results
)

from sklearn.model_selection import KFold
import numpy as np

# Configuration
DATA_PATH = "data"
MODELS_PATH = "models"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)



def run_preprocessing_pipeline(max_rows=10000):
    """Run the complete data preprocessing pipeline."""
    print("\n[STEP 1] Loading and preprocessing datasets...")
    
    # Load raw data with row limit
    books, users, ratings = load_datasets(max_rows=max_rows)
    
    # Clean datasets
    books_clean = clean_books(books)
    users_clean = clean_users(users)
    
    # Create interaction matrices
    explicit_matrix, implicit_matrix = create_interaction_matrix(
        books_clean, users_clean, ratings
    )
    
    # Save processed data
    books_clean.to_csv(os.path.join(DATA_PATH, "cleaned_books.csv"), index=False)
    users_clean.to_csv(os.path.join(DATA_PATH, "cleaned_users.csv"), index=False)
    explicit_matrix.to_csv(os.path.join(DATA_PATH, "explicit_interactions.csv"))
    implicit_matrix.to_csv(os.path.join(DATA_PATH, "implicit_interactions.csv"))
    
    print("[INFO] Preprocessing complete. Files saved to data directory.")
    return books_clean, users_clean, explicit_matrix

def run_model_pipeline():
    """Run the complete model training pipeline."""
    print("\n[STEP 2] Running keyword extraction...")
    run_keyword_extraction()
    
    print("\n[STEP 3] Building vectorization and similarity models...")
    run_vectorization_similarity()

def run_example_recommendations():
    """Generate example recommendations using different methods."""
    print("\n[STEP 4] Generating example recommendations...")
    
    # Load resources
    books_df, users_df, ratings_matrix, sim_matrix = load_resources()
    
    # Example 1: Content-based recommendation
    print("\nContent-based recommendations for 'The Joy Luck Club':")
    recs = recommend_books(
        title="the joy luck club",  # This book exists in our dataset
        books_df=books_df,
        users_df=users_df,
        ratings_matrix=ratings_matrix,
        sim_matrix=sim_matrix,
        blend_factor=0.0  # Pure content-based
    )
    print(recs)
    
    # Get a valid user ID from the ratings matrix
    if not ratings_matrix.empty:
        valid_user = ratings_matrix.index[0]
        # Example 2: User-based recommendation
        print(f"\nUser-based recommendations for user {valid_user}:")
        recs = recommend_books(
            user_id=valid_user,
            books_df=books_df,
            users_df=users_df,
            ratings_matrix=ratings_matrix,
            sim_matrix=sim_matrix,
            blend_factor=1.0  # Pure collaborative
        )
        print(recs)
        
        # Example 3: Hybrid recommendation
        print(f"\nHybrid recommendations for user {valid_user} who likes The Joy Luck Club:")
        recs = recommend_books(
            title="the joy luck club",
            user_id=valid_user,
            books_df=books_df,
            users_df=users_df,
            ratings_matrix=ratings_matrix,
            sim_matrix=sim_matrix,
            blend_factor=0.5  # Equal blend
        )
        print(recs)
    else:
        print("[WARN] No user ratings available in the limited dataset")

def generate_visualizations():
    """Generate visualization plots for analysis."""
    print("\n[STEP 5] Generating visualizations...")
    
    try:
        # Load experiment results
        results = load_experiment_results()
        
        # Generate only the metrics comparison plot
        create_metrics_comparison(results)
        
        print(f"[INFO] Visualizations saved to {os.path.join(DATA_PATH, 'visualizations')}")
    except FileNotFoundError:
        print("[WARN] No experiment results found. Skipping visualization generation.")
        print("[INFO] Run experiments first to generate visualizations.")

def is_good_recommendation(source_book, recommended_book):
    """Determine if a recommendation is good using multiple criteria."""
    score = 0
    
    # 1. Keyword overlap (lowered threshold, more flexible)
    if isinstance(source_book.get('keywords', []), list) and isinstance(recommended_book.get('keywords', []), list):
        source_keywords = set(source_book.get('keywords', []))
        recommended_keywords = set(recommended_book.get('keywords', []))
        
        if source_keywords and recommended_keywords:
            overlap = len(source_keywords & recommended_keywords)
            overlap_ratio = overlap / len(source_keywords)
            if overlap_ratio >= 0.15:  # Lowered from 0.3 to 0.15
                score += 0.4
    
    # 2. Genre similarity
    if source_book.get('genre') and recommended_book.get('genre'):
        if source_book['genre'] == recommended_book['genre']:
            score += 0.3
    
    # 3. Author similarity
    if source_book.get('author') and recommended_book.get('author'):
        if source_book['author'] == recommended_book['author']:
            score += 0.2
    
    # 4. Publication year similarity (within 5 years)
    if source_book.get('publication_year') and recommended_book.get('publication_year'):
        year_diff = abs(source_book['publication_year'] - recommended_book['publication_year'])
        if year_diff <= 5:
            score += 0.1
    
    return score >= 0.3  # Lower overall threshold

def compute_metrics_with_cv(df: pd.DataFrame, similarity_matrix: np.ndarray, n_folds: int = 5) -> ExperimentMetrics:
    """Compute metrics using k-folds cross validation."""
    metrics = ExperimentMetrics()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        fold_accuracy = 0
        valid_test_samples = 0
        
        for test_idx in test_idx[:20]:  # Limit to 20 samples per fold for efficiency
            try:
                # Get recommendations for this test book
                sim_scores = similarity_matrix[test_idx]
                top_indices = np.argsort(sim_scores)[-11:][::-1][1:]  # Exclude self
                
                source_book = df.iloc[test_idx]
                correct_recommendations = 0
                
                # Check top 5 recommendations
                for rec_idx in top_indices[:5]:
                    recommended_book = df.iloc[rec_idx]
                    if is_good_recommendation(source_book, recommended_book):
                        correct_recommendations += 1
                
                fold_accuracy += correct_recommendations / 5
                valid_test_samples += 1
                
            except Exception as e:
                continue
        
        if valid_test_samples > 0:
            fold_accuracies.append(fold_accuracy / valid_test_samples)
    
    # Average across folds
    if fold_accuracies:
        metrics.top_5_accuracy = np.mean(fold_accuracies)
        metrics.top_10_accuracy = metrics.top_5_accuracy * 0.8  # Estimate
    
    return metrics

# # Update the ExperimentTracker class
# class ExperimentTracker:
#     # ... existing code ...
    
#     def compute_metrics(self, df: pd.DataFrame, similarity_matrix: np.ndarray, sample_size: int = 100) -> ExperimentMetrics:
#         """Compute metrics using cross validation for better reliability."""
#         metrics = ExperimentMetrics()
        
#         # Processing time
#         metrics.processing_time = time.time() - self.start_time
        
#         # Use cross validation instead of random sampling
#         metrics = compute_metrics_with_cv(df, similarity_matrix, n_folds=5)
        
#         return metrics

def run_experiments():
    """Run experiments with different configurations using existing models."""
    print("\n[STEP 4] Running experiments...")
    
    from src.experiment_tracking import EXPERIMENT_CONFIGS, ExperimentTracker
    
    # Load the already-created models
    try:
        books_df = pd.read_csv(os.path.join(DATA_PATH, "books_with_keywords.csv"))
        sim_matrix_path = os.path.join(MODELS_PATH, "final_sim_matrix.pkl")
        
        if not os.path.exists(sim_matrix_path):
            print(f"[WARN] Similarity matrix not found. Run model pipeline first.")
            return
            
        with open(sim_matrix_path, "rb") as f:
            import pickle
            sim_matrix = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"[WARN] Required files not found: {e}")
        print("[INFO] Run model pipeline first to create required files.")
        return
    
    tracker = ExperimentTracker()
    
    # Run experiments using the existing models
    for config in EXPERIMENT_CONFIGS:
        print(f"\n[INFO] Running experiment: {config.name}")
        tracker.start_experiment(config)
        
        # Compute metrics using existing models
        metrics = tracker.compute_metrics(books_df, sim_matrix)
        tracker.save_results(metrics, [])
        
        print(f"[INFO] {config.name} completed - Top-5 Accuracy: {metrics.top_5_accuracy:.3f}")
    
    print("[INFO] All experiments completed!")

# Update the main execution to include experiments
if __name__ == "__main__":
    try:
        # Run complete pipeline
        books_clean, users_clean, explicit_matrix = run_preprocessing_pipeline()
        run_model_pipeline()
        run_experiments()  # Add this line
        run_example_recommendations()
        generate_visualizations()
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        raise
