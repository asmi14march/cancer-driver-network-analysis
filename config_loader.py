"""
Configuration Loader

Loads and validates configuration from YAML file.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for cancer driver analysis."""
    
    def __init__(self, config_path='config.yaml'):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key, default=None):
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'data.mafs.raw_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path=None):
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Output path (defaults to original config path)
        """
        output_path = Path(output_path) if output_path else self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __getitem__(self, key):
        """Allow dict-like access."""
        return self.get(key)
    
    def __setitem__(self, key, value):
        """Allow dict-like assignment."""
        self.set(key, value)


# Global config instance
_config = None


def load_config(config_path='config.yaml'):
    """
    Load global configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config
    _config = Config(config_path)
    return _config


def get_config():
    """
    Get global configuration instance.
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


if __name__ == '__main__':
    # Demo usage
    config = load_config()
    
    print("Cancer Driver Network Analysis - Configuration")
    print("="*50)
    print(f"\nData directories:")
    print(f"  MAF raw: {config.get('data.mafs.raw_dir')}")
    print(f"  Networks: {config.get('data.networks.dir')}")
    
    print(f"\nEmbedding parameters:")
    print(f"  Dimensions: {config.get('embedding.node2vec.dimensions')}")
    print(f"  Walk length: {config.get('embedding.node2vec.walk_length')}")
    print(f"  Num walks: {config.get('embedding.node2vec.num_walks')}")
    
    print(f"\nClustering:")
    print(f"  Method: kmeans")
    print(f"  N clusters: {config.get('clustering.kmeans.n_clusters')}")
    
    print(f"\nEvaluation:")
    print(f"  Precision@K values: {config.get('evaluation.precision_at_k')}")
