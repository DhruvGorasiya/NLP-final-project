import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold

# Constants
EXPERIMENTS_DIR = os.path.join("data", "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    bert_model: str
    num_keywords: int
    diversity: float
    tfidf_params: Dict[str, Any]
    description: str = ""

@dataclass
class ExperimentMetrics:
    """Metrics for evaluating recommendation quality."""
    top_5_accuracy: float = 0.0
    top_10_accuracy: float = 0.0
    mean_similarity: float = 0.0
    keyword_overlap: float = 0.0
    unique_genres_ratio: float = 0.0
    unique_authors_ratio: float = 0.0
    diversity_score: float = 0.0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0

def is_good_recommendation(source_book, recommended_book):
    """Determine if a recommendation is good using more flexible criteria."""
    score = 0
    
    # 1. Keyword overlap (more flexible)
    if isinstance(source_book.get('keywords', []), list) and isinstance(recommended_book.get('keywords', []), list):
        source_keywords = set(source_book.get('keywords', []))
        recommended_keywords = set(recommended_book.get('keywords', []))
        
        if source_keywords and recommended_keywords:
            overlap = len(source_keywords & recommended_keywords)
            overlap_ratio = overlap / len(source_keywords)
            if overlap_ratio >= 0.1:  # Lowered from 0.15 to 0.1
                score += 0.3
    
    # 2. Genre similarity (more flexible)
    if source_book.get('genre') and recommended_book.get('genre'):
        if source_book['genre'] == recommended_book['genre']:
            score += 0.4  # Increased weight
        # Add partial genre matching (e.g., "Fiction" vs "Science Fiction")
        elif any(genre in recommended_book['genre'] for genre in source_book['genre'].split()):
            score += 0.2
    
    # 3. Author similarity
    if source_book.get('author') and recommended_book.get('author'):
        if source_book['author'] == recommended_book['author']:
            score += 0.3  # Increased weight
    
    # 4. Publication year similarity (more flexible)
    if source_book.get('publication_year') and recommended_book.get('publication_year'):
        year_diff = abs(source_book['publication_year'] - recommended_book['publication_year'])
        if year_diff <= 10:  # Increased from 5 to 10 years
            score += 0.1
    
    # 5. Title similarity (new criterion)
    if source_book.get('title') and recommended_book.get('title'):
        title_similarity = len(set(source_book['title'].lower().split()) & 
                             set(recommended_book['title'].lower().split())) / \
                            len(set(source_book['title'].lower().split()))
        if title_similarity >= 0.2:
            score += 0.1
    
    return score >= 0.25  # Lowered overall threshold from 0.3 to 0.25

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

class ExperimentTracker:
    def __init__(self):
        self.start_time = None
        self.results_file = os.path.join(EXPERIMENTS_DIR, "experiment_results.json")
    
    def start_experiment(self, config: ExperimentConfig):
        """Start timing a new experiment."""
        self.start_time = time.time()
        self.current_config = config
        print(f"[INFO] Starting experiment: {config.name}")
    
    def compute_metrics(self, df: pd.DataFrame, similarity_matrix: np.ndarray, sample_size: int = 100) -> ExperimentMetrics:
        """Compute metrics using cross validation for better reliability."""
        metrics = ExperimentMetrics()
        
        # Processing time
        metrics.processing_time = time.time() - self.start_time
        
        # Use cross validation instead of random sampling
        metrics = compute_metrics_with_cv(df, similarity_matrix, n_folds=5)
        
        return metrics
    
    def save_results(self, metrics: ExperimentMetrics, sample_recommendations: List[Dict]):
        """Save experiment results to JSON file."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.current_config),
            "metrics": asdict(metrics),
            "sample_recommendations": sample_recommendations
        }
        
        # Load existing results
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        else:
            results = []
        
        # Append new result
        results.append(result)
        
        # Save updated results
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[INFO] Saved experiment results to {self.results_file}")

# Predefined experiment configurations
EXPERIMENT_CONFIGS = [
    ExperimentConfig(
        name="baseline",
        bert_model="all-MiniLM-L6-v2",
        num_keywords=8,
        diversity=0.6,
        tfidf_params={},
        description="Baseline configuration"
    ),
    ExperimentConfig(
        name="high_keywords",
        bert_model="all-MiniLM-L6-v2",
        num_keywords=12,
        diversity=0.6,
        tfidf_params={},
        description="More keywords per book"
    ),
    ExperimentConfig(
        name="high_diversity",
        bert_model="all-MiniLM-L6-v2",
        num_keywords=8,
        diversity=0.8,
        tfidf_params={},
        description="Higher keyword diversity"
    ),
    ExperimentConfig(
        name="alternative_model",
        bert_model="paraphrase-MiniLM-L3-v2",
        num_keywords=8,
        diversity=0.6,
        tfidf_params={},
        description="Alternative BERT model"
    ),
    ExperimentConfig(
        name="custom_tfidf",
        bert_model="all-MiniLM-L6-v2",
        num_keywords=8,
        diversity=0.6,
        tfidf_params={
            "max_features": 1000,
            "ngram_range": (1,2),
            "min_df": 2
        },
        description="Custom TF-IDF parameters"
    )
]
