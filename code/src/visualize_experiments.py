import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

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

def create_kfold_cross_validation_plot(results: List[Dict]):
    """Create k-fold cross validation visualization to address over-fitting/under-fitting."""
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('K-Fold Cross Validation Analysis - Over-fitting/Under-fitting Detection', 
                 fontsize=16, fontweight='bold')
    
    # Extract fold-wise metrics if available
    fold_metrics = {}
    for result in results:
        config_name = result['config']['name']
        if 'fold_metrics' in result:
            fold_metrics[config_name] = result['fold_metrics']
    
    if fold_metrics:
        # Plot 1: Fold-wise accuracy variation
        ax1 = axes[0, 0]
        for config_name, folds in fold_metrics.items():
            fold_numbers = list(range(1, len(folds) + 1))
            accuracies = [fold.get('top_5_accuracy', 0) for fold in folds]
            ax1.plot(fold_numbers, accuracies, marker='o', label=config_name, linewidth=2)
        
        ax1.set_xlabel('Fold Number')
        ax1.set_ylabel('Top-5 Accuracy')
        ax1.set_title('Accuracy Variation Across Folds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation across folds (stability measure)
        ax2 = axes[0, 1]
        config_names = []
        std_devs = []
        for config_name, folds in fold_metrics.items():
            accuracies = [fold.get('top_5_accuracy', 0) for fold in folds]
            std_dev = np.std(accuracies)
            config_names.append(config_name)
            std_devs.append(std_dev)
        
        bars = ax2.bar(config_names, std_devs, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Model Stability (Lower = More Stable)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, std in zip(bars, std_devs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{std:.3f}', ha='center', va='bottom')
        
        # Plot 3: Mean vs Standard Deviation (over-fitting detection)
        ax3 = axes[1, 0]
        means = []
        for config_name, folds in fold_metrics.items():
            accuracies = [fold.get('top_5_accuracy', 0) for fold in folds]
            mean_acc = np.mean(accuracies)
            means.append(mean_acc)
        
        # Color code by stability
        colors = ['green' if std < 0.05 else 'orange' if std < 0.1 else 'red' 
                 for std in std_devs]
        
        scatter = ax3.scatter(means, std_devs, c=colors, s=100, alpha=0.7)
        ax3.set_xlabel('Mean Accuracy')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_title('Over-fitting Detection\n(Green=Stable, Orange=Moderate, Red=Unstable)')
        ax3.grid(True, alpha=0.3)
        
        # Add configuration labels
        for i, config_name in enumerate(config_names):
            ax3.annotate(config_name, (means[i], std_devs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 4: Learning curves (if available)
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Learning Curves\n(Add fold-wise training/validation metrics\nfor complete over-fitting analysis)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Learning Curves')
        ax4.axis('off')
        
    else:
        # If no fold metrics, show placeholder
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'K-Fold metrics not available\nRun experiments with cross-validation enabled', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'kfold_cross_validation_analysis.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_ablation_study_plot(results: List[Dict]):
    """Create comprehensive ablation study visualization with 10+ configurations."""
    plt.figure(figsize=(18, 12))
    
    # Extract configuration parameters and metrics
    ablation_data = []
    for result in results:
        config = result['config']
        metrics = result['metrics']
        
        # Extract key parameters for ablation study
        ablation_data.append({
            'config_name': config['name'],
            'description': config['description'],
            'top_5_accuracy': metrics.get('top_5_accuracy', 0),
            'top_10_accuracy': metrics.get('top_10_accuracy', 0),
            'processing_time': metrics.get('processing_time', 0),
            'vectorization_method': config.get('vectorization_method', 'tfidf'),
            'similarity_metric': config.get('similarity_metric', 'cosine'),
            'keyword_weight': config.get('keyword_weight', 1.0),
            'blend_factor': config.get('blend_factor', 0.5)
        })
    
    df = pd.DataFrame(ablation_data)
    
    # Create comprehensive ablation study visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Comprehensive Ablation Study - Impact of Different Configurations', 
                 fontsize=18, fontweight='bold')
    
    # Plot 1: Accuracy by vectorization method
    ax1 = axes[0, 0]
    if 'vectorization_method' in df.columns:
        method_acc = df.groupby('vectorization_method')['top_5_accuracy'].mean()
        method_acc.plot(kind='bar', ax=ax1, color='lightcoral', alpha=0.8)
        ax1.set_title('Accuracy by Vectorization Method')
        ax1.set_ylabel('Top-5 Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy by similarity metric
    ax2 = axes[0, 1]
    if 'similarity_metric' in df.columns:
        sim_acc = df.groupby('similarity_metric')['top_5_accuracy'].mean()
        sim_acc.plot(kind='bar', ax=ax2, color='lightblue', alpha=0.8)
        ax2.set_title('Accuracy by Similarity Metric')
        ax2.set_ylabel('Top-5 Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy vs Processing Time
    ax3 = axes[0, 2]
    scatter = ax3.scatter(df['processing_time'], df['top_5_accuracy'], 
                          c=df['top_5_accuracy'], cmap='viridis', s=100, alpha=0.7)
    ax3.set_xlabel('Processing Time (seconds)')
    ax3.set_ylabel('Top-5 Accuracy')
    ax3.set_title('Accuracy vs Processing Time')
    ax3.grid(True, alpha=0.3)
    
    # Add configuration labels
    for i, config_name in enumerate(df['config_name']):
        ax3.annotate(config_name, (df.iloc[i]['processing_time'], df.iloc[i]['top_5_accuracy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Blend factor impact
    ax4 = axes[1, 0]
    if 'blend_factor' in df.columns:
        blend_acc = df.groupby('blend_factor')['top_5_accuracy'].mean()
        blend_acc.plot(kind='line', ax=ax4, marker='o', linewidth=2, markersize=8)
        ax4.set_xlabel('Blend Factor (0=Content, 1=Collaborative)')
        ax4.set_ylabel('Top-5 Accuracy')
        ax4.set_title('Impact of Hybrid Blend Factor')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Keyword weight impact
    ax5 = axes[1, 1]
    if 'keyword_weight' in df.columns:
        keyword_acc = df.groupby('keyword_weight')['top_5_accuracy'].mean()
        keyword_acc.plot(kind='line', ax=ax5, marker='s', linewidth=2, markersize=8, color='orange')
        ax5.set_title('Impact of Keyword Weight')
        ax5.set_ylabel('Top-5 Accuracy')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Configuration ranking
    ax6 = axes[1, 2]
    # Sort by accuracy and create ranking
    df_sorted = df.sort_values('top_5_accuracy', ascending=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax6.barh(y_pos, df_sorted['top_5_accuracy'], color='lightgreen', alpha=0.8)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(df_sorted['config_name'])
    ax6.set_xlabel('Top-5 Accuracy')
    ax6.set_title('Configuration Ranking by Performance')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax6.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'comprehensive_ablation_study.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_extreme_error_analysis(results: List[Dict]):
    """Create extreme error analysis visualization for insightful error investigation."""
    plt.figure(figsize=(16, 12))
    
    # Extract error-related metrics
    error_data = []
    for result in results:
        config = result['config']
        metrics = result['metrics']
        
        # Calculate error metrics
        accuracy = metrics.get('top_5_accuracy', 0)
        error_rate = 1 - accuracy
        
        error_data.append({
            'config_name': config['name'],
            'accuracy': accuracy,
            'error_rate': error_rate,
            'processing_time': metrics.get('processing_time', 0),
            'vectorization_method': config.get('vectorization_method', 'tfidf'),
            'similarity_metric': config.get('similarity_metric', 'cosine')
        })
    
    df = pd.DataFrame(error_data)
    
    # Create extreme error analysis visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Extreme Error Analysis - Understanding Model Failures', 
                 fontsize=18, fontweight='bold')
    
    # Plot 1: Error rate distribution
    ax1 = axes[0, 0]
    error_bins = np.linspace(0, 1, 11)
    ax1.hist(df['error_rate'], bins=error_bins, color='lightcoral', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Error Rate')
    ax1.set_ylabel('Number of Configurations')
    ax1.set_title('Distribution of Error Rates')
    ax1.grid(True, alpha=0.3)
    
    # Add mean line
    mean_error = df['error_rate'].mean()
    ax1.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                label=f'Mean Error: {mean_error:.3f}')
    ax1.legend()
    
    # Plot 2: Error rate vs Processing time
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['processing_time'], df['error_rate'], 
                          c=df['error_rate'], cmap='Reds', s=100, alpha=0.7)
    ax2.set_xlabel('Processing Time (seconds)')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('Error Rate vs Processing Time')
    ax2.grid(True, alpha=0.3)
    
    # Add configuration labels for high error configurations
    high_error_threshold = df['error_rate'].quantile(0.75)
    for i, row in df.iterrows():
        if row['error_rate'] > high_error_threshold:
            ax2.annotate(row['config_name'], (row['processing_time'], row['error_rate']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 3: Error rate by vectorization method
    ax3 = axes[1, 0]
    if 'vectorization_method' in df.columns:
        method_error = df.groupby('vectorization_method')['error_rate'].mean()
        method_error.plot(kind='bar', ax=ax3, color='lightcoral', alpha=0.8)
        ax3.set_title('Error Rate by Vectorization Method')
        ax3.set_ylabel('Error Rate')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(method_error):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 4: Error rate by similarity metric
    ax4 = axes[1, 1]
    if 'similarity_metric' in df.columns:
        sim_error = df.groupby('similarity_metric')['error_rate'].mean()
        sim_error.plot(kind='bar', ax=ax4, color='lightblue', alpha=0.8)
        ax4.set_title('Error Rate by Similarity Metric')
        ax4.set_ylabel('Error Rate')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(sim_error):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'extreme_error_analysis.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def generate_all_visualizations():
    """Generate comprehensive visualizations including new evaluation strategies."""
    print("[INFO] Loading experiment results...")
    results = load_experiment_results()
    
    print("[INFO] Creating metrics comparison plots...")
    create_metrics_comparison(results)
    
    print("[INFO] Creating k-fold cross validation analysis...")
    create_kfold_cross_validation_plot(results)
    
    print("[INFO] Creating comprehensive ablation study...")
    create_ablation_study_plot(results)
    
    print("[INFO] Creating extreme error analysis...")
    create_extreme_error_analysis(results)
    
    print(f"[INFO] All visualizations saved to {VISUALIZATIONS_DIR}")
    print("[INFO] Generated visualizations:")
    print("  - top5_accuracy_comparison.png")
    print("  - kfold_cross_validation_analysis.png")
    print("  - comprehensive_ablation_study.png")
    print("  - extreme_error_analysis.png")

if __name__ == "__main__":
    generate_all_visualizations()