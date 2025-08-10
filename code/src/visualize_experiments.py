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
    """Create comparison plots for different metrics across experiments."""
    # Extract metrics into DataFrame
    metrics_data = []
    for result in results:
        metrics = result['metrics']
        config = result['config']
        metrics_data.append({
            'experiment': config['name'],
            'description': config['description'],
            'Top-5 Accuracy': metrics['top_5_accuracy'],
            'Keyword Overlap': metrics['keyword_overlap'],
            'Diversity Score': metrics['diversity_score'],
            'Processing Time': metrics['processing_time']
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Create subplot with 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['Top-5 Accuracy', 'Keyword Overlap', 'Diversity Score', 'Processing Time']
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes.flat)):
        sns.barplot(data=df, x='experiment', y=metric, ax=ax)
        ax.set_title(metric)
        
        # Set x-ticks explicitly
        ax.set_xticks(range(len(df['experiment'])))
        ax.set_xticklabels(df['experiment'], rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.suptitle("Experiment Metrics Comparison", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'metrics_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_recommendation_heatmap(results: List[Dict]):
    """Create similarity heatmap for sample recommendations."""
    # Get first experiment's sample recommendations
    samples = results[0]['sample_recommendations']
    
    # Create similarity matrix for visualization
    titles = [rec['book'] for rec in samples]
    n = len(titles)
    sim_matrix = np.zeros((n, n))
    
    # Fill diagonal with 1.0 (self-similarity)
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Fill similarity values
    for i, rec in enumerate(samples):
        for j, r in enumerate(rec['recommendations']):
            sim_matrix[i, j] = r['similarity']
            sim_matrix[j, i] = r['similarity']  # Make it symmetric
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(sim_matrix, xticklabels=titles, yticklabels=titles, 
                annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Book Similarity Heatmap (Sample)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'similarity_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def create_parameter_impact_plot(results: List[Dict]):
    """Create visualization showing impact of different parameters."""
    param_data = []
    for result in results:
        config = result['config']
        metrics = result['metrics']
        param_data.append({
            'experiment': config['name'],
            'num_keywords': config['num_keywords'],
            'diversity': config['diversity'],
            'accuracy': metrics['top_5_accuracy'],
            'processing_time': metrics['processing_time']
        })
    
    df = pd.DataFrame(param_data)
    
    # Create scatter plot with size indicating processing time
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['num_keywords'], df['accuracy'], 
                         s=df['processing_time']*100,  # Scale size for visibility
                         c=df['diversity'], cmap='viridis',
                         alpha=0.6)
    
    # Add labels for each point
    for idx, row in df.iterrows():
        plt.annotate(row['experiment'], 
                    (row['num_keywords'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(scatter, label='Diversity Parameter')
    plt.xlabel('Number of Keywords')
    plt.ylabel('Top-5 Accuracy')
    plt.title('Parameter Impact on Model Performance')
    
    # Add legend for bubble size
    legend_elements = [plt.scatter([], [], s=t*100, c='gray', alpha=0.3,
                                 label=f'{t:.1f}s')
                      for t in [min(df['processing_time']), 
                              max(df['processing_time'])]]
    plt.legend(handles=legend_elements, 
              labels=['Min Processing Time', 'Max Processing Time'],
              title='Bubble Size', loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'parameter_impact.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations for the experiments."""
    print("[INFO] Loading experiment results...")
    results = load_experiment_results()
    
    print("[INFO] Creating metrics comparison plots...")
    create_metrics_comparison(results)
    
    print("[INFO] Creating recommendation heatmap...")
    create_recommendation_heatmap(results)
    
    print("[INFO] Creating parameter impact visualization...")
    create_parameter_impact_plot(results)
    
    print(f"[INFO] All visualizations saved to {VISUALIZATIONS_DIR}")

if __name__ == "__main__":
    generate_all_visualizations()