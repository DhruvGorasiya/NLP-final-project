import os
import pandas as pd
from src.keyword_extraction import run_keyword_extraction
from src.vectorization_similarity import run_vectorization_similarity, build_tfidf_matrix, compute_cosine_similarity
from src.experiment_tracking import EXPERIMENT_CONFIGS, ExperimentTracker
from sklearn.feature_extraction.text import TfidfVectorizer

def run_all_experiments():
    """Run all predefined experiments and track results."""
    
    print("[INFO] Starting experiment suite...")
    
    for config in EXPERIMENT_CONFIGS:
        print(f"\n{'='*50}")
        print(f"Running experiment: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*50}\n")
        
        # Initialize tracker
        tracker = ExperimentTracker()
        tracker.start_experiment(config)
        
        try:
            # Run keyword extraction
            df, output_file = run_keyword_extraction(config)
            
            # Run vectorization with custom TF-IDF params if specified
            print("[INFO] Building TF-IDF matrix...")
            if config.tfidf_params:
                print("[INFO] Using custom TF-IDF parameters...")
                vectorizer = TfidfVectorizer(**config.tfidf_params)
            else:
                vectorizer = TfidfVectorizer()
            
            docs = [" ".join(keywords) for keywords in df["keywords"]]
            tfidf_matrix = vectorizer.fit_transform(docs)
            sim_matrix = compute_cosine_similarity(tfidf_matrix)
                
            # Compute and save metrics
            metrics = tracker.compute_metrics(df, sim_matrix)
            
            # Get sample recommendations
            sample_recs = []
            for i in range(min(5, len(df))):
                sim_scores = sim_matrix[i]
                top_indices = sim_scores.argsort()[-6:][::-1][1:]  # Top 5 excluding self
                sample_recs.append({
                    "book": df.iloc[i]["Book-Title"],
                    "recommendations": [
                        {
                            "title": df.iloc[j]["Book-Title"],
                            "author": df.iloc[j]["Book-Author"],
                            "similarity": float(sim_scores[j])
                        }
                        for j in top_indices
                    ]
                })
            
            # Save results
            tracker.save_results(metrics, sample_recs)
            
        except Exception as e:
            print(f"[ERROR] Experiment failed: {str(e)}")
            continue
    
    print("\n[INFO] All experiments completed!")

if __name__ == "__main__":
    run_all_experiments()
