"""
Unit tests for GNN embedding module.
"""

import unittest
import numpy as np
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.embedding import Node2VecEmbedding


class TestNode2VecEmbedding(unittest.TestCase):
    """Test cases for Node2Vec embedding generation."""
    
    def setUp(self):
        """Create a simple test graph."""
        self.G = nx.karate_club_graph()
        self.embedder = Node2VecEmbedding(
            dimensions=8,
            walk_length=10,
            num_walks=20,
            workers=1
        )
    
    def test_initialization(self):
        """Test Node2Vec initialization."""
        self.assertEqual(self.embedder.dimensions, 8)
        self.assertEqual(self.embedder.walk_length, 10)
        self.assertEqual(self.embedder.num_walks, 20)
        self.assertEqual(self.embedder.p, 1)
        self.assertEqual(self.embedder.q, 1)
    
    def test_random_walk_generation(self):
        """Test random walk generation."""
        walks = self.embedder._generate_walks(self.G)
        
        # Should have num_walks per node
        expected_walks = self.G.number_of_nodes() * self.embedder.num_walks
        self.assertEqual(len(walks), expected_walks)
        
        # Each walk should have correct length
        self.assertTrue(all(len(walk) == self.embedder.walk_length for walk in walks))
    
    def test_fit(self):
        """Test embedding fitting."""
        self.embedder.fit(self.G)
        embeddings = self.embedder.get_embeddings()
        
        # Should have embeddings for all nodes
        self.assertEqual(len(embeddings), self.G.number_of_nodes())
        
        # Each embedding should have correct dimensions
        for node, emb in embeddings.items():
            self.assertEqual(len(emb), self.embedder.dimensions)
    
    def test_save_load_embeddings(self):
        """Test saving and loading embeddings."""
        import tempfile
        
        self.embedder.fit(self.G)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name
        
        try:
            # Save embeddings
            self.embedder.save_embeddings_txt(temp_path)
            
            # Check file exists and has content
            self.assertTrue(Path(temp_path).exists())
            
            # Load and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), self.G.number_of_nodes())
            
        finally:
            Path(temp_path).unlink()
    
    def test_biased_random_walk(self):
        """Test biased random walk with different p, q values."""
        embedder_biased = Node2VecEmbedding(
            dimensions=8,
            walk_length=10,
            num_walks=20,
            p=0.5,
            q=2.0,
            workers=1
        )
        
        embedder_biased.fit(self.G)
        embeddings = embedder_biased.get_embeddings()
        
        self.assertEqual(len(embeddings), self.G.number_of_nodes())


class TestDeepWalk(unittest.TestCase):
    """Test cases for DeepWalk (unbiased Node2Vec)."""
    
    def setUp(self):
        """Create a simple test graph."""
        self.G = nx.karate_club_graph()
    
    def test_deepwalk_is_unbiased_node2vec(self):
        """Test that DeepWalk is equivalent to Node2Vec with p=1, q=1."""
        embedder = Node2VecEmbedding(
            dimensions=8,
            walk_length=10,
            num_walks=20,
            p=1.0,
            q=1.0,
            workers=1
        )
        
        embedder.fit(self.G)
        embeddings = embedder.get_embeddings()
        
        self.assertEqual(len(embeddings), self.G.number_of_nodes())
        self.assertTrue(all(len(emb) == 8 for emb in embeddings.values()))


if __name__ == '__main__':
    unittest.main()
