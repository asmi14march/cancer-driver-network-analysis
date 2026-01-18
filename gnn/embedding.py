"""
Graph Neural Network Embedding Module

This module implements graph embedding techniques for cancer driver network analysis.
Includes Node2Vec, DeepWalk, and network-based approaches.
"""

import numpy as np
import networkx as nx
from pathlib import Path
from gensim.models import Word2Vec


class Node2VecEmbedding:
    """
    Node2Vec embedding for cancer driver networks.
    Uses biased random walks to learn node representations.
    
    Reference: Grover & Leskovec (2016). "node2vec: Scalable Feature Learning for Networks"
    """
    
    def __init__(self, dimensions=128, walk_length=80, num_walks=10, 
                 p=1.0, q=1.0, workers=4, window=10):
        """
        Initialize Node2Vec parameters.
        
        Args:
            dimensions: Embedding dimension
            walk_length: Length of random walk
            num_walks: Number of walks per node
            p: Return parameter (controls likelihood of returning to previous node)
            q: In-out parameter (controls exploration vs exploitation)
            workers: Number of parallel workers
            window: Context window size for Skip-gram
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.window = window
        self.model = None
        
    def _generate_walks(self, G):
        """Generate random walks for all nodes"""
        walks = []
        nodes = list(G.nodes())
        
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(G, node)
                walks.append([str(n) for n in walk])
        
        return walks
    
    def _random_walk(self, G, start_node):
        """
        Perform a single biased random walk.
        
        Args:
            G: NetworkX graph
            start_node: Starting node
            
        Returns:
            List of nodes in walk
        """
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            neighbors = list(G.neighbors(cur))
            
            if len(neighbors) == 0:
                break
            
            # Biased random walk based on p and q parameters
            if len(walk) == 1:
                walk.append(np.random.choice(neighbors))
            else:
                prev = walk[-2]
                probs = []
                for neighbor in neighbors:
                    if neighbor == prev:
                        probs.append(1/self.p)
                    elif G.has_edge(neighbor, prev):
                        probs.append(1)
                    else:
                        probs.append(1/self.q)
                
                probs = np.array(probs)
                probs = probs / probs.sum()
                walk.append(np.random.choice(neighbors, p=probs))
        
        return walk
    
    def fit(self, G):
        """
        Train Node2Vec on graph.
        
        Args:
            G: NetworkX graph
        """
        print(f"Generating random walks...")
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        walks = self._generate_walks(G)
        print(f"  Generated {len(walks)} walks")
        
        print(f"Training Word2Vec model...")
        self.model = Word2Vec(
            walks, 
            vector_size=self.dimensions, 
            window=self.window, 
            min_count=0, 
            workers=self.workers, 
            sg=1,  # Skip-gram
            hs=0,  # Negative sampling
            negative=5
        )
        print(f"  Model trained: {len(self.model.wv)} node embeddings")
        
    def get_embeddings(self):
        """
        Get node embeddings as dictionary.
        
        Returns:
            Dictionary mapping node names to embedding vectors
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        embeddings = {}
        for node in self.model.wv.index_to_key:
            embeddings[node] = self.model.wv[node]
        
        return embeddings
    
    def save(self, filepath):
        """Save model to file"""
        if self.model:
            self.model.save(str(filepath))
            print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        self.model = Word2Vec.load(str(filepath))
        print(f"Model loaded from {filepath}")
    
    def save_embeddings_txt(self, filepath):
        """
        Save embeddings in text format (gene vector format).
        
        Args:
            filepath: Output file path
        """
        embeddings = self.get_embeddings()
        with open(filepath, "w") as f:
            for node, vector in embeddings.items():
                vector_str = ' '.join(map(str, vector))
                f.write(f"{node} {vector_str}\n")
        print(f"Embeddings saved to {filepath}")


class DeepWalkEmbedding(Node2VecEmbedding):
    """
    DeepWalk embedding (special case of Node2Vec with p=1, q=1).
    Performs unbiased random walks.
    
    Reference: Perozzi et al. (2014). "DeepWalk: Online Learning of Social Representations"
    """
    
    def __init__(self, dimensions=128, walk_length=80, num_walks=10, 
                 workers=4, window=10):
        """Initialize DeepWalk (Node2Vec with p=1, q=1)"""
        super().__init__(dimensions, walk_length, num_walks, 
                        p=1.0, q=1.0, workers=workers, window=window)


class GraphEmbedding:
    """
    General graph embedding wrapper supporting multiple algorithms.
    """
    
    def __init__(self, method='node2vec', **kwargs):
        """
        Initialize graph embedding.
        
        Args:
            method: Embedding method ('node2vec', 'deepwalk')
            **kwargs: Method-specific parameters
        """
        self.method = method
        
        if method == 'node2vec':
            self.embedder = Node2VecEmbedding(**kwargs)
        elif method == 'deepwalk':
            self.embedder = DeepWalkEmbedding(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, G):
        """Fit embedding model"""
        self.embedder.fit(G)
    
    def transform(self, nodes=None):
        """Get embeddings for nodes"""
        embeddings = self.embedder.get_embeddings()
        
        if nodes is None:
            return embeddings
        
        return {node: embeddings[str(node)] for node in nodes if str(node) in embeddings}
    
    def fit_transform(self, G, nodes=None):
        """Fit and transform in one step"""
        self.fit(G)
        return self.transform(nodes)


def load_network(network_path):
    """
    Load network from edge list file.
    
    Args:
        network_path: Path to edge list file
        
    Returns:
        NetworkX graph
    """
    G = nx.read_edgelist(network_path)
    print(f"Loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def generate_embeddings(network_path, output_path, dimensions=128, 
                       walk_length=80, num_walks=10, method='node2vec'):
    """
    Complete pipeline to generate embeddings from network file.
    
    Args:
        network_path: Path to input network edge list
        output_path: Path to save embeddings
        dimensions: Embedding dimensions
        walk_length: Random walk length
        num_walks: Number of walks per node
        method: Embedding method
    """
    # Load network
    G = load_network(network_path)
    
    # Generate embeddings
    embedder = GraphEmbedding(
        method=method,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks
    )
    embedder.fit(G)
    
    # Save embeddings
    embedder.embedder.save_embeddings_txt(output_path)
    
    return embedder


# Standalone script functionality
if __name__ == "__main__":
    import sys
    
    # Default paths
    NETWORK_PATH = "../data/networks/network.edgelist"
    EMBEDDING_OUT = "../data/networks/embeddings.txt"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        NETWORK_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        EMBEDDING_OUT = sys.argv[2]
    
    print("="*60)
    print("Node2Vec Embedding Generation")
    print("="*60)
    print(f"Input network: {NETWORK_PATH}")
    print(f"Output embeddings: {EMBEDDING_OUT}")
    print("")
    
    # Generate embeddings
    generate_embeddings(
        network_path=NETWORK_PATH,
        output_path=EMBEDDING_OUT,
        dimensions=128,
        walk_length=80,
        num_walks=10,
        method='node2vec'
    )
    
    print("")
    print("="*60)
    print("Embeddings generated successfully!")
    print("="*60)
