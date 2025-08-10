import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top-5 Accuracy', 'Keyword Overlap', 
                       'Diversity Score', 'Processing Time (s)')
    )
    
    metrics = ['Top-5 Accuracy', 'Keyword Overlap', 'Diversity Score', 'Processing Time']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for metric, pos in zip(metrics, positions):
        fig.add_trace(
            go.Bar(
                x=df['experiment'],
                y=df[metric],
                text=df[metric].round(3),
                textposition='auto',
                name=metric
            ),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Experiment Metrics Comparison"
    )
    
    # Save interactive HTML
    fig.write_html(os.path.join(VISUALIZATIONS_DIR, 'metrics_comparison.html'))
    
    # Save static image for paper
    fig.write_image(os.path.join(VISUALIZATIONS_DIR, 'metrics_comparison.png'))

def create_recommendation_heatmap(results: List[Dict]):
    """Create similarity heatmap for sample recommendations."""
    # Get first experiment's sample recommendations
    samples = results[0]['sample_recommendations']
    
    # Create similarity matrix for visualization
    titles = [rec['book'] for rec in samples]
    sim_matrix = np.zeros((len(titles), len(titles)))
    
    for i, rec in enumerate(samples):
        for j, r in enumerate(rec['recommendations']):
            sim_matrix[i][j+1] = r['similarity']
            sim_matrix[j+1][i] = r['similarity']  # Make it symmetric
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(sim_matrix, xticklabels=titles, yticklabels=titles, 
                annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Book Similarity Heatmap (Sample)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'similarity_heatmap.png'))
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
    
    # Create bubble plot
    fig = px.scatter(df, x='num_keywords', y='accuracy',
                    size='processing_time', color='diversity',
                    hover_name='experiment',
                    labels={
                        'num_keywords': 'Number of Keywords',
                        'accuracy': 'Top-5 Accuracy',
                        'diversity': 'Diversity Parameter'
                    },
                    title='Parameter Impact on Model Performance')
    
    # Save interactive HTML
    fig.write_html(os.path.join(VISUALIZATIONS_DIR, 'parameter_impact.html'))
    
    # Save static image for paper
    fig.write_image(os.path.join(VISUALIZATIONS_DIR, 'parameter_impact.png'))

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
