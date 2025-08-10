import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from sklearn.metrics import pairwise_distances

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

class ExperimentTracker:
    def __init__(self):
        self.start_time = None
        self.results_file = os.path.join(EXPERIMENTS_DIR, "experiment_results.json")
    
    def start_experiment(self, config: ExperimentConfig):
        """Start timing a new experiment."""
        self.start_time = time.time()
        self.current_config = config
        print(f"[INFO] Starting experiment: {config.name}")
    
    def compute_metrics(self, 
                       df: pd.DataFrame,
                       similarity_matrix: np.ndarray,
                       sample_size: int = 100) -> ExperimentMetrics:
        """Compute all evaluation metrics."""
        metrics = ExperimentMetrics()
        
        # Processing time
        metrics.processing_time = time.time() - self.start_time
        
        # Sample random books for evaluation
        sample_indices = np.random.choice(len(df), min(sample_size, len(df)), replace=False)
        
        # Compute metrics for sampled books
        for idx in sample_indices:
            # Get top 10 recommendations for this book
            sim_scores = similarity_matrix[idx]
            top_indices = np.argsort(sim_scores)[-11:][::-1][1:]  # Exclude self
            
            # Genre overlap (if available)
            if 'Genre' in df.columns:
                book_genres = set(str(df.iloc[idx]['Genre']).split(','))
                rec_genres = set()
                rec_authors = set()
                
                for rec_idx in top_indices[:5]:  # Top 5
                    rec_genres.update(str(df.iloc[rec_idx]['Genre']).split(','))
                    rec_authors.add(df.iloc[rec_idx]['Book-Author'])
                
                metrics.top_5_accuracy += len(book_genres & rec_genres) / len(book_genres)
                metrics.unique_genres_ratio += len(rec_genres) / (5 * len(book_genres))
                metrics.unique_authors_ratio += len(rec_authors) / 5
            
            # Keyword overlap
            book_keywords = set(df.iloc[idx]['keywords'])
            rec_keywords = set()
            for rec_idx in top_indices[:5]:
                rec_keywords.update(df.iloc[rec_idx]['keywords'])
            metrics.keyword_overlap += len(book_keywords & rec_keywords) / len(book_keywords)
            
            # Recommendation diversity (average pairwise distance)
            rec_vectors = similarity_matrix[top_indices[:5]]
            diversity = np.mean(pairwise_distances(rec_vectors))
            metrics.diversity_score += diversity
        
        # Average metrics across samples
        metrics.top_5_accuracy /= len(sample_indices)
        metrics.keyword_overlap /= len(sample_indices)
        metrics.unique_genres_ratio /= len(sample_indices)
        metrics.unique_authors_ratio /= len(sample_indices)
        metrics.diversity_score /= len(sample_indices)
        
        # Overall similarity score
        metrics.mean_similarity = np.mean(similarity_matrix[similarity_matrix > 0])
        
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
