#!/usr/bin/env python3
"""
Command-Line Interface for Cancer Driver Network Analysis

Usage:
    python cli.py embed --network NETWORK_FILE --output OUTPUT_FILE [options]
    python cli.py cluster --embeddings EMBEDDINGS_FILE --output OUTPUT_FILE [options]
    python cli.py evaluate --predictions PRED_FILE --known KNOWN_FILE [options]
    python cli.py pipeline --maf MAF_FILE --network NETWORK_FILE [options]
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from gnn.embedding import generate_embeddings
from gnn.clustering import cluster_genes
from gnn.correlation_analysis import analyze_driver_correlations


def cmd_embed(args):
    """Generate network embeddings."""
    print(f"Generating embeddings from {args.network}...")
    
    generate_embeddings(
        network_path=args.network,
        output_path=args.output,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        workers=args.workers
    )
    
    print(f"Embeddings saved to {args.output}")


def cmd_cluster(args):
    """Cluster genes based on embeddings."""
    print(f"Clustering genes from {args.embeddings}...")
    
    cluster_genes(
        embeddings_path=args.embeddings,
        output_path=args.output,
        n_clusters=args.n_clusters,
        method=args.method
    )
    
    print(f"Clusters saved to {args.output}")


def cmd_correlate(args):
    """Analyze correlation with known drivers."""
    print(f"Analyzing driver correlations...")
    
    analyze_driver_correlations(
        embeddings_path=args.embeddings,
        candidates_path=args.candidates,
        known_drivers_path=args.known_drivers,
        output_path=args.output,
        method=args.method
    )
    
    print(f"Correlation results saved to {args.output}")


def cmd_evaluate(args):
    """Evaluate predictions against known drivers."""
    import pandas as pd
    import numpy as np
    
    print(f"Evaluating predictions from {args.predictions}...")
    
    # Load predictions
    predictions = pd.read_csv(args.predictions, sep='\t')
    if 'Gene' in predictions.columns:
        pred_genes = predictions['Gene'].tolist()
    else:
        pred_genes = predictions.iloc[:, 0].tolist()
    
    # Load known drivers
    with open(args.known, 'r') as f:
        known_drivers = set([line.strip() for line in f])
    
    # Calculate precision at K
    k_values = args.k_values if args.k_values else [5, 10, 20, 50, 100]
    
    print("\nEvaluation Results:")
    print("="*50)
    
    for k in k_values:
        if k > len(pred_genes):
            continue
        
        top_k = set(pred_genes[:k])
        tp = len(top_k.intersection(known_drivers))
        precision = tp / k
        
        print(f"Precision@{k}: {precision:.4f} ({tp}/{k})")
    
    print("="*50)


def cmd_pipeline(args):
    """Run complete analysis pipeline."""
    print("Running complete cancer driver analysis pipeline...")
    print("="*60)
    
    # Step 1: Generate embeddings
    if not args.skip_embed:
        print("\nStep 1: Generating network embeddings...")
        embeddings_path = Path(args.output_dir) / "embeddings.txt"
        generate_embeddings(
            network_path=args.network,
            output_path=embeddings_path,
            dimensions=args.dimensions,
            num_walks=args.num_walks
        )
    else:
        embeddings_path = Path(args.embeddings)
    
    # Step 2: Cluster genes
    if not args.skip_cluster:
        print("\nStep 2: Clustering genes...")
        clusters_path = Path(args.output_dir) / "clusters.csv"
        cluster_genes(
            embeddings_path=embeddings_path,
            output_path=clusters_path,
            n_clusters=args.n_clusters
        )
    
    # Step 3: Analyze correlations if known drivers provided
    if args.known_drivers:
        print("\nStep 3: Analyzing driver correlations...")
        corr_path = Path(args.output_dir) / "driver_correlations.csv"
        analyze_driver_correlations(
            embeddings_path=embeddings_path,
            candidates_path=args.candidates,
            known_drivers_path=args.known_drivers,
            output_path=corr_path
        )
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print(f"Results saved to: {args.output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Cancer Driver Network Analysis - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate network embeddings')
    embed_parser.add_argument('--network', required=True, help='Network file path')
    embed_parser.add_argument('--output', required=True, help='Output embeddings file')
    embed_parser.add_argument('--dimensions', type=int, default=128, help='Embedding dimensions')
    embed_parser.add_argument('--walk-length', type=int, default=80, help='Random walk length')
    embed_parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per node')
    embed_parser.add_argument('--p', type=float, default=1.0, help='Return parameter')
    embed_parser.add_argument('--q', type=float, default=1.0, help='In-out parameter')
    embed_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    embed_parser.set_defaults(func=cmd_embed)
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Cluster genes by embeddings')
    cluster_parser.add_argument('--embeddings', required=True, help='Embeddings file path')
    cluster_parser.add_argument('--output', required=True, help='Output clusters file')
    cluster_parser.add_argument('--n-clusters', type=int, default=10, help='Number of clusters')
    cluster_parser.add_argument('--method', choices=['kmeans', 'spectral'], default='kmeans', help='Clustering method')
    cluster_parser.set_defaults(func=cmd_cluster)
    
    # Correlate command
    correlate_parser = subparsers.add_parser('correlate', help='Analyze driver correlations')
    correlate_parser.add_argument('--embeddings', required=True, help='Embeddings file path')
    correlate_parser.add_argument('--candidates', required=True, help='Candidate genes file')
    correlate_parser.add_argument('--known-drivers', required=True, help='Known drivers file')
    correlate_parser.add_argument('--output', required=True, help='Output correlations file')
    correlate_parser.add_argument('--method', choices=['pearson', 'spearman', 'cosine'], 
                                 default='cosine', help='Correlation method')
    correlate_parser.set_defaults(func=cmd_correlate)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    eval_parser.add_argument('--predictions', required=True, help='Predictions file')
    eval_parser.add_argument('--known', required=True, help='Known drivers file')
    eval_parser.add_argument('--k-values', nargs='+', type=int, help='K values for precision@k')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--network', required=True, help='Network file path')
    pipeline_parser.add_argument('--output-dir', default='./results', help='Output directory')
    pipeline_parser.add_argument('--dimensions', type=int, default=128, help='Embedding dimensions')
    pipeline_parser.add_argument('--num-walks', type=int, default=10, help='Number of walks')
    pipeline_parser.add_argument('--n-clusters', type=int, default=10, help='Number of clusters')
    pipeline_parser.add_argument('--candidates', help='Candidate genes file')
    pipeline_parser.add_argument('--known-drivers', help='Known drivers file')
    pipeline_parser.add_argument('--embeddings', help='Pre-computed embeddings')
    pipeline_parser.add_argument('--skip-embed', action='store_true', help='Skip embedding generation')
    pipeline_parser.add_argument('--skip-cluster', action='store_true', help='Skip clustering')
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
