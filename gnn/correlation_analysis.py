"""
Correlation Analysis Module

This module performs correlation analysis on cancer driver networks and embeddings.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path


class CorrelationAnalysis:
    """
    Class for analyzing correlations in cancer driver networks.
    """
    
    def __init__(self, method='pearson'):
        """
        Initialize the correlation analysis.
        
        Args:
            method: Correlation method ('pearson', 'spearman')
        """
        self.method = method
        
    def compute_correlation_matrix(self, data):
        """
        Compute correlation matrix for the given data.
        
        Args:
            data: Input data matrix (samples x features) or DataFrame
            
        Returns:
            correlation_matrix: Correlation matrix
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if self.method == 'pearson':
            correlation_matrix = np.corrcoef(data.T)
        elif self.method == 'spearman':
            from scipy.stats import spearmanr
            correlation_matrix, _ = spearmanr(data, axis=0)
        
        return correlation_matrix
    
    def identify_significant_correlations(self, correlation_matrix, threshold=0.5):
        """
        Identify significant correlations above a threshold.
        
        Args:
            correlation_matrix: Correlation matrix
            threshold: Significance threshold
            
        Returns:
            significant_pairs: List of significantly correlated pairs
        """
        significant_pairs = []
        n = correlation_matrix.shape[0]
        
        for i in range(n):
            for j in range(i+1, n):
                if abs(correlation_matrix[i, j]) >= threshold:
                    significant_pairs.append((i, j, correlation_matrix[i, j]))
        
        return significant_pairs
    
    def analyze_network_properties(self, adjacency_matrix):
        """
        Analyze basic network properties.
        
        Args:
            adjacency_matrix: Network adjacency matrix
            
        Returns:
            properties: Dictionary of network properties
        """
        properties = {}
        properties['num_nodes'] = adjacency_matrix.shape[0]
        properties['num_edges'] = np.sum(adjacency_matrix > 0) / 2
        properties['density'] = properties['num_edges'] / (properties['num_nodes'] * (properties['num_nodes'] - 1) / 2)
        
        return properties
    
    def correlate_with_drivers(self, embeddings_df, known_drivers):
        """
        Correlate candidate genes with known driver genes based on embeddings.
        
        Args:
            embeddings_df: DataFrame with gene embeddings (gene as first column)
            known_drivers: Set of known driver gene names
            
        Returns:
            DataFrame with gene-driver correlations
        """
        genes = embeddings_df.iloc[:, 0].values
        X = embeddings_df.iloc[:, 1:].values
        
        results = []
        
        for i, gene in enumerate(genes):
            if gene in known_drivers:
                continue
            
            for j, driver in enumerate(genes):
                if driver in known_drivers:
                    if self.method == 'pearson':
                        corr, p_value = pearsonr(X[i], X[j])
                    elif self.method == 'spearman':
                        corr, p_value = spearmanr(X[i], X[j])
                    
                    results.append({
                        'gene': gene,
                        'driver': driver,
                        'correlation': corr,
                        'p_value': p_value
                    })
        
        df = pd.DataFrame(results)
        return df.sort_values('correlation', ascending=False)


def load_embeddings(filepath):
    """
    Load embeddings from text file.
    
    Args:
        filepath: Path to embeddings file
        
    Returns:
        DataFrame with embeddings
    """
    embeddings = pd.read_csv(filepath, sep=" ", header=None)
    print(f"Loaded embeddings: {len(embeddings)} genes, {embeddings.shape[1]-1} dimensions")
    return embeddings


def load_known_drivers(filepath):
    """
    Load known driver genes from file.
    
    Args:
        filepath: Path to driver genes file (one gene per line)
        
    Returns:
        Set of driver gene names
    """
    drivers = set(pd.read_csv(filepath, header=None)[0].values)
    print(f"Loaded {len(drivers)} known driver genes")
    return drivers


def analyze_driver_correlations(embeddings_path, drivers_path, output_path, 
                                top_k=20, method='pearson'):
    """
    Complete pipeline to analyze correlations with known drivers.
    
    Args:
        embeddings_path: Path to embeddings file
        drivers_path: Path to known drivers file
        output_path: Path to save results
        top_k: Number of top correlations to save
        method: Correlation method
    """
    print("="*60)
    print("Driver Correlation Analysis")
    print("="*60)
    
    # Load data
    embeddings = load_embeddings(embeddings_path)
    drivers = load_known_drivers(drivers_path)
    
    # Analyze correlations
    analyzer = CorrelationAnalysis(method=method)
    results = analyzer.correlate_with_drivers(embeddings, drivers)
    
    # Save top results
    top_results = results.head(top_k)
    top_results.to_csv(output_path, index=False)
    
    print(f"\nTop {top_k} correlations:")
    print(top_results.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    
    return results


# Standalone script functionality
if __name__ == "__main__":
    import sys
    
    # Default paths
    EMBEDDINGS_PATH = "../data/networks/embeddings.txt"
    DRIVERS_PATH = "../data/evaluation/reference/known_cancer_drivers.txt"
    OUTPUT_PATH = "../data/evaluation/top_correlations.csv"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        EMBEDDINGS_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        DRIVERS_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        OUTPUT_PATH = sys.argv[3]
    
    # Run analysis
    analyze_driver_correlations(
        embeddings_path=EMBEDDINGS_PATH,
        drivers_path=DRIVERS_PATH,
        output_path=OUTPUT_PATH,
        top_k=20,
        method='pearson'
    )
    
    print("\n" + "="*60)
    print("Correlation analysis complete!")
    print("="*60)
