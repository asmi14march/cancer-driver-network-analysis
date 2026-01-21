"""
Unit tests for clustering module.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.clustering import GraphClustering


class TestGraphClustering(unittest.TestCase):
    """Test cases for graph clustering."""
    
    def setUp(self):
        """Create sample embeddings."""
        np.random.seed(42)
        # Create 3 clusters of 10 points each
        cluster1 = np.random.randn(10, 8) + [5, 5, 5, 5, 0, 0, 0, 0]
        cluster2 = np.random.randn(10, 8) + [-5, -5, -5, -5, 0, 0, 0, 0]
        cluster3 = np.random.randn(10, 8) + [0, 0, 0, 0, 5, 5, 5, 5]
        
        self.embeddings = np.vstack([cluster1, cluster2, cluster3])
        self.n_samples = 30
        self.n_features = 8
        self.true_clusters = 3
    
    def test_kmeans_clustering(self):
        """Test K-means clustering."""
        clusterer = GraphClustering(n_clusters=3, method='kmeans')
        labels = clusterer.fit_predict(self.embeddings)
        
        # Should return correct number of labels
        self.assertEqual(len(labels), self.n_samples)
        
        # Should have 3 unique clusters
        unique_labels = set(labels)
        self.assertEqual(len(unique_labels), 3)
        
        # Labels should be in range [0, n_clusters)
        self.assertTrue(all(0 <= label < 3 for label in labels))
    
    def test_spectral_clustering(self):
        """Test spectral clustering."""
        clusterer = GraphClustering(n_clusters=3, method='spectral')
        labels = clusterer.fit_predict(self.embeddings)
        
        self.assertEqual(len(labels), self.n_samples)
        self.assertEqual(len(set(labels)), 3)
    
    def test_evaluate(self):
        """Test clustering evaluation metrics."""
        clusterer = GraphClustering(n_clusters=3, method='kmeans')
        metrics = clusterer.evaluate(self.embeddings)
        
        # Should return metrics dictionary
        self.assertIn('silhouette_score', metrics)
        self.assertIn('davies_bouldin_score', metrics)
        
        # Silhouette score should be between -1 and 1
        self.assertTrue(-1 <= metrics['silhouette_score'] <= 1)
        
        # Davies-Bouldin score should be positive
        self.assertTrue(metrics['davies_bouldin_score'] >= 0)
    
    def test_different_cluster_sizes(self):
        """Test clustering with different number of clusters."""
        for n_clusters in [2, 3, 5]:
            clusterer = GraphClustering(n_clusters=n_clusters, method='kmeans')
            labels = clusterer.fit_predict(self.embeddings)
            
            unique_labels = set(labels)
            self.assertEqual(len(unique_labels), n_clusters)
    
    def test_random_state(self):
        """Test reproducibility with random state."""
        clusterer1 = GraphClustering(n_clusters=3, method='kmeans', random_state=42)
        clusterer2 = GraphClustering(n_clusters=3, method='kmeans', random_state=42)
        
        labels1 = clusterer1.fit_predict(self.embeddings)
        labels2 = clusterer2.fit_predict(self.embeddings)
        
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_dataframe_input(self):
        """Test clustering with pandas DataFrame input."""
        df = pd.DataFrame(self.embeddings)
        clusterer = GraphClustering(n_clusters=3, method='kmeans')
        labels = clusterer.fit_predict(df)
        
        self.assertEqual(len(labels), self.n_samples)


if __name__ == '__main__':
    unittest.main()
