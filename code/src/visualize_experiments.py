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
    """Load all experiment results from JSON file and filter out broken experiments."""
    results_file = os.path.join(EXPERIMENTS_DIR, "experiment_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"No experiment results found at {results_file}")
    
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Filter out experiments with 0.0 accuracy (broken experiments)
    valid_results = []
    for result in all_results:
        metrics = result.get('metrics', {})
        if metrics.get('top_5_accuracy', 0.0) > 0.0:  # Only include experiments with meaningful accuracy
            valid_results.append(result)
    
    print(f"[INFO] Loaded {len(all_results)} total experiments, {len(valid_results)} valid experiments")
    
    if not valid_results:
        raise ValueError("No valid experiments found with meaningful accuracy scores")
    
    return valid_results

def create_metrics_comparison(results: List[Dict]):
    """Create comparison plots for different metrics across experiments."""
    # Extract metrics into DataFrame
    metrics_data = []
    for result in results:
        metrics = result.get('metrics', {})
        config = result.get('config', {})
        
        # Validate required fields
        if not all(key in metrics for key in ['top_5_accuracy', 'top_10_accuracy', 'keyword_overlap', 'processing_time']):
            print(f"[WARN] Skipping experiment {config.get('name', 'unknown')} - missing metrics")
            continue
            
        metrics_data.append({
            'experiment': config.get('name', 'unknown'),
            'description': config.get('description', ''),
            'Top-5 Accuracy': metrics['top_5_accuracy'],
            'Top-10 Accuracy': metrics['top_10_accuracy'],
            'Keyword Overlap': metrics['keyword_overlap'],
            'Processing Time': metrics['processing_time']
        })
    
    if not metrics_data:
        raise ValueError("No valid metrics data found for visualization")
    
    df = pd.DataFrame(metrics_data)
    print(f"[INFO] Creating metrics comparison for {len(df)} experiments")
    
    # Create subplot with 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['Top-5 Accuracy', 'Top-10 Accuracy', 'Keyword Overlap', 'Processing Time']
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes.flat)):
        # Create bar plot with better styling
        bars = sns.barplot(data=df, x='experiment', y=metric, ax=ax, hue='experiment', legend=False)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Experiment', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        
        # Set x-ticks explicitly with better formatting
        ax.set_xticks(range(len(df['experiment'])))
        ax.set_xticklabels(df['experiment'], rotation=45, ha='right', fontsize=9)
        
        # Add value labels on top of bars with better formatting
        for i, v in enumerate(df[metric]):
            if metric == 'Processing Time':
                label = f'{v:.1f}s'
            else:
                label = f'{v:.3f}'
            ax.text(i, v + max(df[metric])*0.01, label, ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Experiment Metrics Comparison", y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'metrics_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved metrics comparison plot to {os.path.join(VISUALIZATIONS_DIR, 'metrics_comparison.png')}")

def create_recommendation_heatmap(results: List[Dict]):
    """Create similarity heatmap for sample recommendations."""
    # Get first experiment's sample recommendations
    if not results or 'sample_recommendations' not in results[0]:
        print("[WARN] No sample recommendations found, skipping heatmap")
        return
    
    samples = results[0]['sample_recommendations']
    if not samples:
        print("[WARN] Empty sample recommendations, skipping heatmap")
        return
    
    # Create similarity matrix for visualization
    titles = [rec.get('book', f'Book_{i}') for i, rec in enumerate(samples)]
    n = len(titles)
    sim_matrix = np.zeros((n, n))
    
    # Fill diagonal with 1.0 (self-similarity)
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Fill similarity values correctly
    for i, rec in enumerate(samples):
        recommendations = rec.get('recommendations', [])
        for j, r in enumerate(recommendations):
            if j < n:  # Ensure we don't go out of bounds
                sim_matrix[i, j] = r.get('similarity', 0.0)
                sim_matrix[j, i] = r.get('similarity', 0.0)  # Make it symmetric
    
    # Create heatmap with better styling
    plt.figure(figsize=(14, 10))
    
    # Truncate long titles for better display
    short_titles = [title[:30] + '...' if len(title) > 30 else title for title in titles]
    
    heatmap = sns.heatmap(sim_matrix, xticklabels=short_titles, yticklabels=short_titles, 
                          annot=True, fmt='.2f', cmap='YlOrRd', 
                          cbar_kws={'label': 'Similarity Score'})
    
    plt.title('Book Similarity Heatmap (Sample Recommendations)', fontsize=14, fontweight='bold')
    plt.xlabel('Recommended Books', fontsize=12)
    plt.ylabel('Source Books', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'similarity_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved similarity heatmap to {os.path.join(VISUALIZATIONS_DIR, 'similarity_heatmap.png')}")

def create_parameter_impact_plot(results: List[Dict]):
    """Create visualization showing impact of different parameters."""
    param_data = []
    for result in results:
        config = result.get('config', {})
        metrics = result.get('metrics', {})
        
        # Validate required fields
        required_config = ['name', 'num_keywords', 'diversity']
        required_metrics = ['top_5_accuracy', 'processing_time']
        
        if not all(key in config for key in required_config):
            print(f"[WARN] Skipping experiment {config.get('name', 'unknown')} - missing config fields")
            continue
            
        if not all(key in metrics for key in required_metrics):
            print(f"[WARN] Skipping experiment {config.get('name', 'unknown')} - missing metrics fields")
            continue
        
        param_data.append({
            'experiment': config['name'],
            'num_keywords': config['num_keywords'],
            'diversity': config['diversity'],
            'accuracy': metrics['top_5_accuracy'],
            'processing_time': metrics['processing_time']
        })
    
    if not param_data:
        raise ValueError("No valid parameter data found for visualization")
    
    df = pd.DataFrame(param_data)
    print(f"[INFO] Creating parameter impact plot for {len(df)} experiments")
    
    # Create scatter plot with size indicating processing time
    plt.figure(figsize=(12, 8))
    
    # Scale bubble sizes for better visibility
    min_size = 100
    max_size = 1000
    sizes = min_size + (df['processing_time'] - df['processing_time'].min()) / (df['processing_time'].max() - df['processing_time'].min()) * (max_size - min_size)
    
    scatter = plt.scatter(df['num_keywords'], df['accuracy'], 
                         s=sizes, c=df['diversity'], cmap='viridis',
                         alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add labels for each point with better positioning
    for idx, row in df.iterrows():
        plt.annotate(row['experiment'], 
                    (row['num_keywords'], row['accuracy']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.colorbar(scatter, label='Diversity Parameter', shrink=0.8)
    plt.xlabel('Number of Keywords', fontsize=12, fontweight='bold')
    plt.ylabel('Top-5 Accuracy', fontsize=12, fontweight='bold')
    plt.title('Parameter Impact on Model Performance', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add legend for bubble size
    min_time = df['processing_time'].min()
    max_time = df['processing_time'].max()
    legend_elements = [
        plt.scatter([], [], s=min_size, c='gray', alpha=0.7, edgecolors='black', linewidth=1,
                   label=f'{min_time:.1f}s'),
        plt.scatter([], [], s=max_size, c='gray', alpha=0.7, edgecolors='black', linewidth=1,
                   label=f'{max_time:.1f}s')
    ]
    plt.legend(handles=legend_elements, 
              title='Processing Time', loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'parameter_impact.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved parameter impact plot to {os.path.join(VISUALIZATIONS_DIR, 'parameter_impact.png')}")

def create_accuracy_trend_plot(results: List[Dict]):
    """Create a plot showing accuracy trends across experiments."""
    # Extract accuracy data with deduplication (take latest version of each experiment)
    experiment_dict = {}  # Store latest version of each experiment
    
    for result in results:
        config = result.get('config', {})
        metrics = result.get('metrics', {})
        timestamp = result.get('timestamp', '')
        experiment_name = config.get('name', 'unknown')
        
        if 'top_5_accuracy' in metrics and 'top_10_accuracy' in metrics:
            # Keep the version with the latest timestamp (most recent run)
            if experiment_name not in experiment_dict or timestamp > experiment_dict[experiment_name]['timestamp']:
                experiment_dict[experiment_name] = {
                    'experiment': experiment_name,
                    'top_5_accuracy': metrics['top_5_accuracy'],
                    'top_10_accuracy': metrics['top_10_accuracy'],
                    'timestamp': timestamp
                }
    
    accuracy_data = [data for data in experiment_dict.values()]
    
    if not accuracy_data:
        print("[WARN] No accuracy data found for trend plot")
        return
    
    df = pd.DataFrame(accuracy_data)
    print(f"[INFO] Creating accuracy trends plot for {len(df)} unique experiments")
    print(f"[INFO] Experiments included: {list(df['experiment'])}")
    
    # Create trend plot
    plt.figure(figsize=(12, 6))
    
    x = range(len(df))
    plt.plot(x, df['top_5_accuracy'], 'o-', label='Top-5 Accuracy', linewidth=2, markersize=8)
    plt.plot(x, df['top_10_accuracy'], 's-', label='Top-10 Accuracy', linewidth=2, markersize=8)
    
    # Add experiment labels
    plt.xticks(x, df['experiment'], rotation=45, ha='right')
    
    plt.xlabel('Experiments', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    plt.title('Accuracy Trends Across Experiments', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (top5, top10) in enumerate(zip(df['top_5_accuracy'], df['top_10_accuracy'])):
        plt.annotate(f'{top5:.3f}', (i, top5), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
        plt.annotate(f'{top10:.3f}', (i, top10), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'accuracy_trends.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved accuracy trends plot to {os.path.join(VISUALIZATIONS_DIR, 'accuracy_trends.png')}")

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
    
    print("[INFO] Creating accuracy trends plot...")
    create_accuracy_trend_plot(results)
    
    print(f"[INFO] All visualizations saved to {VISUALIZATIONS_DIR}")

if __name__ == "__main__":
    generate_all_visualizations()