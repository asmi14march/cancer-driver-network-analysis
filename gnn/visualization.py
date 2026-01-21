"""
Visualization Module

Provides visualization functions for cancer driver network analysis:
- Network graphs
- Embedding scatter plots
- Precision-recall curves
- Clustering visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_network(G, node_colors=None, node_sizes=None, title="Gene Network", 
                 layout='spring', figsize=(12, 10), save_path=None):
    """
    Plot a network graph.
    
    Args:
        G: NetworkX graph
        node_colors: Dict or list of node colors
        node_sizes: Dict or list of node sizes
        title: Plot title
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        figsize: Figure size tuple
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Prepare node colors
    if node_colors is None:
        node_colors = ['lightblue'] * len(G.nodes())
    elif isinstance(node_colors, dict):
        node_colors = [node_colors.get(node, 'lightblue') for node in G.nodes()]
    
    # Prepare node sizes
    if node_sizes is None:
        node_sizes = [300] * len(G.nodes())
    elif isinstance(node_sizes, dict):
        node_sizes = [node_sizes.get(node, 300) for node in G.nodes()]
    
    # Draw network
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embeddings_2d(embeddings, labels=None, method='tsne', title="Gene Embeddings",
                        figsize=(10, 8), save_path=None):
    """
    Plot embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings: Dict of gene->embedding or numpy array
        labels: Optional cluster labels
        method: Reduction method ('tsne', 'pca')
        title: Plot title
        figsize: Figure size tuple
        save_path: Path to save figure
    """
    # Convert to array if dict
    if isinstance(embeddings, dict):
        genes = list(embeddings.keys())
        X = np.array([embeddings[g] for g in genes])
    else:
        genes = None
        X = embeddings
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=figsize)
    
    if labels is not None:
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', 
                            s=100, alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, label='Cluster')
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], s=100, alpha=0.6, edgecolors='black')
    
    # Add gene labels if available
    if genes:
        for i, gene in enumerate(genes):
            plt.annotate(gene, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    
    plt.title(f"{title} ({method.upper()})", fontsize=16, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(precision_values, recall_values, k_values=None,
                                 title="Precision-Recall Curve", figsize=(10, 6),
                                 save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        precision_values: List of precision values
        recall_values: List of recall values
        k_values: Optional K values for annotation
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    plt.plot(recall_values, precision_values, 'b-o', linewidth=2, markersize=8)
    
    # Annotate K values if provided
    if k_values:
        for k, r, p in zip(k_values, recall_values, precision_values):
            plt.annotate(f'K={k}', (r, p), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_at_k(k_values, precision_values, title="Precision@K",
                        figsize=(10, 6), save_path=None):
    """
    Plot precision at different K values.
    
    Args:
        k_values: List of K values
        precision_values: List of precision values
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    plt.bar(range(len(k_values)), precision_values, color='steelblue', alpha=0.7)
    plt.xticks(range(len(k_values)), [f'K={k}' for k in k_values])
    
    # Add value labels on bars
    for i, v in enumerate(precision_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('K (Top K Predictions)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_heatmap(correlation_matrix, genes, title="Gene Correlation Heatmap",
                              figsize=(12, 10), save_path=None):
    """
    Plot correlation heatmap.
    
    Args:
        correlation_matrix: NxN correlation matrix
        genes: List of gene names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(correlation_matrix, xticklabels=genes, yticklabels=genes,
                cmap='coolwarm', center=0, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_cluster_distribution(labels, title="Cluster Distribution",
                               figsize=(8, 6), save_path=None):
    """
    Plot distribution of samples across clusters.
    
    Args:
        labels: Cluster labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.bar(unique_labels, counts, color='coral', alpha=0.7, edgecolor='black')
    
    # Add count labels
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        plt.text(label, count + max(counts)*0.01, str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Genes', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(unique_labels)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_degree_distribution(G, title="Network Degree Distribution",
                              figsize=(10, 6), save_path=None):
    """
    Plot degree distribution of network.
    
    Args:
        G: NetworkX graph
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    degrees = [G.degree(n) for n in G.nodes()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(degrees, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Degree Distribution', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2.boxplot(degrees, vert=True)
    ax2.set_ylabel('Degree', fontsize=12)
    ax2.set_title('Degree Summary', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Demo visualization
    print("Cancer Driver Network Analysis - Visualization Module")
    print("Import this module to use visualization functions")
    print("\nAvailable functions:")
    print("  - plot_network()")
    print("  - plot_embeddings_2d()")
    print("  - plot_precision_recall_curve()")
    print("  - plot_precision_at_k()")
    print("  - plot_correlation_heatmap()")
    print("  - plot_cluster_distribution()")
    print("  - plot_degree_distribution()")
