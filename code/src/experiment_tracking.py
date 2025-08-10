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
        
        try:
            # Processing time
            metrics.processing_time = time.time() - self.start_time
            
            # Sample random books for evaluation
            sample_indices = np.random.choice(len(df), min(sample_size, len(df)), replace=False)
            valid_samples = 0
            
            # Compute metrics for sampled books
            for idx in sample_indices:
                try:
                    # Get top recommendations
                    sim_scores = similarity_matrix[idx]
                    top_indices = np.argsort(sim_scores)[-11:][::-1][1:]  # Exclude self
                    
                    # Calculate accuracy based on keyword overlap
                    if isinstance(df.iloc[idx]['keywords'], list):
                        book_keywords = set(df.iloc[idx]['keywords'])
                        if book_keywords:  # Only process if we have keywords
                            # Calculate individual recommendation accuracy
                            top_5_overlaps = []
                            top_10_overlaps = []
                            
                            # Check top 5 recommendations
                            for rec_idx in top_indices[:5]:
                                if isinstance(df.iloc[rec_idx]['keywords'], list):
                                    rec_keywords = set(df.iloc[rec_idx]['keywords'])
                                    if rec_keywords:
                                        overlap = len(book_keywords & rec_keywords) / len(book_keywords)
                                        top_5_overlaps.append(overlap)
                            
                            # Check top 10 recommendations
                            for rec_idx in top_indices[:10]:
                                if isinstance(df.iloc[rec_idx]['keywords'], list):
                                    rec_keywords = set(df.iloc[rec_idx]['keywords'])
                                    if rec_keywords:
                                        overlap = len(book_keywords & rec_keywords) / len(book_keywords)
                                        top_10_overlaps.append(overlap)
                            
                            # Average the overlaps
                            if top_5_overlaps:
                                metrics.top_5_accuracy += np.mean(top_5_overlaps)
                            if top_10_overlaps:
                                metrics.top_10_accuracy += np.mean(top_10_overlaps)
                            
                            valid_samples += 1
                
                    # Recommendation diversity
                    if len(top_indices) >= 5:
                        rec_vectors = similarity_matrix[top_indices[:5]]
                        if rec_vectors.shape[0] > 1:
                            metrics.diversity_score += np.mean(pairwise_distances(rec_vectors))
                
                except Exception as e:
                    print(f"[WARN] Error processing sample {idx}: {str(e)}")
                    continue
            
            # Average metrics across valid samples
            if valid_samples > 0:
                metrics.top_5_accuracy /= valid_samples
                metrics.top_10_accuracy /= valid_samples
                metrics.diversity_score /= valid_samples
                
                # Overall similarity score
                pos_sim = similarity_matrix[similarity_matrix > 0]
                if len(pos_sim) > 0:
                    metrics.mean_similarity = float(np.mean(pos_sim))
                
                # Calculate overall keyword overlap across all books
                all_keywords = set()
                all_rec_keywords = set()
                for i in range(min(100, len(df))):  # Sample for efficiency
                    if isinstance(df.iloc[i]['keywords'], list):
                        all_keywords.update(df.iloc[i]['keywords'])
                        sim_scores = similarity_matrix[i]
                        top_indices = np.argsort(sim_scores)[-6:][::-1][1:][:5]
                        for rec_idx in top_indices:
                            if isinstance(df.iloc[rec_idx]['keywords'], list):
                                all_rec_keywords.update(df.iloc[rec_idx]['keywords'])
                
                if all_keywords and all_rec_keywords:
                    metrics.keyword_overlap = len(all_keywords & all_rec_keywords) / len(all_keywords)
        
        except Exception as e:
            print(f"[ERROR] Error computing metrics: {str(e)}")
            import traceback
            traceback.print_exc()
        
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
        name="very_high_keywords",
        bert_model="all-MiniLM-L6-v2",
        num_keywords=16,
        diversity=0.6,
        tfidf_params={},
        description="Many keywords per book"
    ),
    ExperimentConfig(
        name="extreme_keywords",
        bert_model="all-MiniLM-L6-v2",
        num_keywords=20,
        diversity=0.6,
        tfidf_params={},
        description="Extreme number of keywords"
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
