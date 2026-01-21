"""
Unit tests for correlation analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.correlation_analysis import CorrelationAnalysis


class TestCorrelationAnalysis(unittest.TestCase):
    """Test cases for correlation analysis."""
    
    def setUp(self):
        """Create sample embeddings and candidate genes."""
        np.random.seed(42)
        
        # Create sample embeddings
        self.genes = ['TP53', 'KRAS', 'EGFR', 'BRAF', 'PIK3CA', 'PTEN', 'NRAS', 'APC']
        self.embeddings = {gene: np.random.randn(8) for gene in self.genes}
        
        # Candidate genes (some overlap with embeddings)
        self.candidates = ['TP53', 'KRAS', 'EGFR', 'NEWGENE1', 'NEWGENE2']
        
        # Known drivers (subset of embeddings)
        self.known_drivers = ['TP53', 'KRAS', 'BRAF', 'PIK3CA']
    
    def test_initialization(self):
        """Test CorrelationAnalysis initialization."""
        analyzer = CorrelationAnalysis(method='pearson')
        self.assertEqual(analyzer.method, 'pearson')
        
        analyzer = CorrelationAnalysis(method='cosine')
        self.assertEqual(analyzer.method, 'cosine')
    
    def test_compute_similarity(self):
        """Test similarity computation between embeddings."""
        analyzer = CorrelationAnalysis(method='cosine')
        
        # Identical vectors should have similarity ~1
        vec1 = np.array([1, 2, 3, 4])
        vec2 = np.array([1, 2, 3, 4])
        sim = analyzer._compute_similarity(vec1, vec2)
        self.assertAlmostEqual(sim, 1.0, places=5)
        
        # Orthogonal vectors should have similarity ~0
        vec1 = np.array([1, 0, 0, 0])
        vec2 = np.array([0, 1, 0, 0])
        sim = analyzer._compute_similarity(vec1, vec2)
        self.assertAlmostEqual(sim, 0.0, places=5)
    
    def test_compute_correlation_matrix(self):
        """Test correlation matrix computation."""
        analyzer = CorrelationAnalysis(method='pearson')
        corr_matrix = analyzer.compute_correlation_matrix(self.embeddings)
        
        # Should be square matrix
        self.assertEqual(corr_matrix.shape[0], len(self.genes))
        self.assertEqual(corr_matrix.shape[1], len(self.genes))
        
        # Diagonal should be 1 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(len(self.genes)))
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)
    
    def test_find_top_correlated(self):
        """Test finding top correlated genes."""
        analyzer = CorrelationAnalysis(method='cosine')
        
        # Create embeddings where some are more similar
        embeddings = {
            'GENE1': np.array([1, 0, 0, 0]),
            'GENE2': np.array([0.9, 0.1, 0, 0]),  # Similar to GENE1
            'GENE3': np.array([0, 1, 0, 0]),
            'GENE4': np.array([0, 0, 1, 0])
        }
        
        top_corr = analyzer.find_top_correlated('GENE1', embeddings, top_k=2)
        
        # Should return top_k results
        self.assertEqual(len(top_corr), 2)
        
        # GENE2 should be most correlated with GENE1
        self.assertEqual(top_corr[0][0], 'GENE2')
    
    def test_correlate_with_drivers(self):
        """Test correlation with known driver genes."""
        analyzer = CorrelationAnalysis(method='cosine')
        
        # Correlate candidates with known drivers
        results = analyzer.correlate_with_drivers(
            self.candidates,
            self.known_drivers,
            self.embeddings
        )
        
        # Should return DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        
        # Should have candidate column
        self.assertIn('candidate', results.columns)
        
        # Should have mean_correlation column
        self.assertIn('mean_correlation', results.columns)
    
    def test_different_correlation_methods(self):
        """Test different correlation methods."""
        methods = ['pearson', 'spearman', 'cosine']
        
        for method in methods:
            analyzer = CorrelationAnalysis(method=method)
            corr_matrix = analyzer.compute_correlation_matrix(self.embeddings)
            
            self.assertEqual(corr_matrix.shape[0], len(self.genes))
            self.assertEqual(corr_matrix.shape[1], len(self.genes))


if __name__ == '__main__':
    unittest.main()
