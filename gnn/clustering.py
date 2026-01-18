"""
Graph Clustering Module

This module implements clustering algorithms for cancer driver network analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from pathlib import Path


class GraphClustering:
    """
    Class for clustering nodes in cancer driver networks.
    """
    
    def __init__(self, n_clusters, method='kmeans', random_state=42):
        """
        Initialize the clustering model.
        
        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'spectral')
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.model = None
        
    def fit_predict(self, embeddings):
        """
        Fit the clustering model and predict cluster labels.
        
        Args:
            embeddings: Node embeddings (numpy array or DataFrame)
            
        Returns:
            labels: Cluster labels for each node
        """
        if isinstance(embeddings, pd.DataFrame):
            embeddings = embeddings.values
            
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.method == 'spectral':
            self.model = SpectralClustering(n_clusters=self.n_clusters, random_state=self.random_state)
        
        labels = self.model.fit_predict(embeddings)
        return labels
    
    def evaluate(self, embeddings, true_labels=None):
        """
        Evaluate clustering quality.
        
        Args:
            embeddings: Node embeddings
            true_labels: Optional ground truth labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        labels = self.fit_predict(embeddings)
        
        metrics = {}
        metrics['silhouette_score'] = silhouette_score(embeddings, labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
        
        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
            metrics['nmi_score'] = normalized_mutual_info_score(true_labels, labels)
        
        return metrics


def load_embeddings(filepath):
    """
    Load embeddings from text file.
    
    Args:
        filepath: Path to embeddings file
        
    Returns:
        genes: Gene names
        embeddings: Embedding matrix
    """
    embeddings = pd.read_csv(filepath, sep=" ", header=None)
    genes = embeddings.iloc[:, 0].values
    X = embeddings.iloc[:, 1:].values
    
    print(f"Loaded embeddings: {len(genes)} genes, {X.shape[1]} dimensions")
    return genes, X


def cluster_genes(embeddings_path, output_path, n_clusters=10, method='kmeans'):
    """
    Complete pipeline to cluster genes based on embeddings.
    
    Args:
        embeddings_path: Path to embeddings file
        output_path: Path to save clustered genes
        n_clusters: Number of clusters
        method: Clustering method
    """
    print("="*60)
    print("Gene Clustering Analysis")
    print("="*60)
    
    # Load embeddings
    genes, X = load_embeddings(embeddings_path)
    
    # Perform clustering
    clusterer = GraphClustering(n_clusters=n_clusters, method=method)
    labels = clusterer.fit_predict(X)
    
    # Create results dataframe
    clustered = pd.DataFrame({
        "gene": genes,
        "cluster": labels
    })
    
    # Save results
    clustered.to_csv(output_path, index=False)
    
    # Display summary
    print(f"\nClustering completed:")
    print(f"  Method: {method}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Genes clustered: {len(genes)}")
    
    print(f"\nCluster sizes:")
    cluster_counts = clustered['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} genes")
    
    # Evaluate clustering quality
    try:
        metrics = clusterer.evaluate(X)
        print(f"\nClustering quality metrics:")
        print(f"  Silhouette score: {metrics['silhouette_score']:.3f}")
        print(f"  Davies-Bouldin score: {metrics['davies_bouldin_score']:.3f}")
    except Exception as e:
        print(f"\nCould not compute quality metrics: {e}")
    
    print(f"\nResults saved to: {output_path}")
    
    return clustered


# Standalone script functionality
if __name__ == "__main__":
    import sys
    
    # Default paths
    EMBEDDINGS_PATH = "../data/networks/embeddings.txt"
    OUTPUT_PATH = "../data/networks/gene_clusters.csv"
    N_CLUSTERS = 10
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        EMBEDDINGS_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        N_CLUSTERS = int(sys.argv[3])
    
    # Run clustering
    cluster_genes(
        embeddings_path=EMBEDDINGS_PATH,
        output_path=OUTPUT_PATH,
        n_clusters=N_CLUSTERS,
        method='kmeans'
    )
    
    print("\n" + "="*60)
    print("Clustering complete!")
    print("="*60)
