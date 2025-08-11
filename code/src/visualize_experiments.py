import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Constants
EXPERIMENTS_DIR = os.path.join("data", "experiments")
VISUALIZATIONS_DIR = os.path.join("data", "visualizations")
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def load_experiment_results() -> List[Dict]:
    """Load all experiment results from JSON file."""
    results_file = os.path.join(EXPERIMENTS_DIR, "experiment_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"No experiment results found at {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def create_metrics_comparison(results: List[Dict]):
    """Create comparison plot for Top-5 Accuracy only."""
    # Extract metrics into DataFrame
    metrics_data = []
    for result in results:
        metrics = result['metrics']
        config = result['config']
        metrics_data.append({
            'experiment': config['name'],
            'description': config['description'],
            'Top-5 Accuracy': metrics['top_5_accuracy']
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Create single plot for Top-5 Accuracy
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    sns.barplot(data=df, x='experiment', y='Top-5 Accuracy', ax=ax)
    ax.set_title('Top-5 Accuracy Comparison')
    
    # Set x-ticks explicitly
    ax.set_xticks(range(len(df['experiment'])))
    ax.set_xticklabels(df['experiment'], rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(df['Top-5 Accuracy']):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.title("Experiment Top-5 Accuracy Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'top5_accuracy_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def generate_all_visualizations():
    """Generate only the essential visualizations."""
    print("[INFO] Loading experiment results...")
    results = load_experiment_results()
    
    print("[INFO] Creating metrics comparison plots...")
    create_metrics_comparison(results)
    
    print(f"[INFO] Visualizations saved to {VISUALIZATIONS_DIR}")

if __name__ == "__main__":
    generate_all_visualizations()