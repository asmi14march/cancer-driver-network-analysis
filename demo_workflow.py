#!/usr/bin/env python3
"""
Demo script to demonstrate the cancer driver network analysis workflow
Creates sample data and runs through the pipeline steps
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def create_sample_maf():
    """Create a sample MAF file for demonstration"""
    print("\n" + "="*60)
    print("Step 1: Creating Sample MAF Data")
    print("="*60)
    
    # Sample gene mutations
    genes = ['TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'NRAS', 'APC', 'CDKN2A', 'RB1']
    samples = [f'TCGA-PATIENT-{i:03d}' for i in range(1, 51)]
    
    data = []
    np.random.seed(42)
    
    for _ in range(200):
        gene = np.random.choice(genes, p=[0.20, 0.15, 0.12, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06, 0.04])
        sample = np.random.choice(samples)
        variant = np.random.choice(['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins'])
        
        data.append({
            'Hugo_Symbol': gene,
            'Tumor_Sample_Barcode': sample,
            'Variant_Classification': variant,
            'Chromosome': np.random.choice(['1', '2', '3', '7', '10', '17']),
            'Start_Position': np.random.randint(1000000, 50000000),
            'Reference_Allele': np.random.choice(['A', 'C', 'G', 'T']),
            'Tumor_Seq_Allele2': np.random.choice(['A', 'C', 'G', 'T'])
        })
    
    maf_df = pd.DataFrame(data)
    
    # Save to file
    output_path = Path('data/mafs/raw/sample_study.maf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maf_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Created sample MAF with {len(maf_df)} mutations")
    print(f"  Genes: {len(maf_df['Hugo_Symbol'].unique())}")
    print(f"  Samples: {len(maf_df['Tumor_Sample_Barcode'].unique())}")
    print(f"  Saved to: {output_path}")
    
    return maf_df

def clean_maf(maf_df):
    """Clean MAF file - filter for protein-altering mutations"""
    print("\n" + "="*60)
    print("Step 2: Cleaning MAF (Filter Protein-Altering Mutations)")
    print("="*60)
    
    # Filter for protein-altering mutations
    protein_altering = [
        'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
        'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins'
    ]
    
    cleaned = maf_df[maf_df['Variant_Classification'].isin(protein_altering)].copy()
    
    print(f"Original mutations: {len(maf_df)}")
    print(f"Protein-altering mutations: {len(cleaned)}")
    print(f"Filtered out: {len(maf_df) - len(cleaned)} ({100*(len(maf_df)-len(cleaned))/len(maf_df):.1f}%)")
    
    # Save cleaned MAF
    output_path = Path('data/mafs/clean/sample_study_cleaned.maf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to: {output_path}")
    
    return cleaned

def maf_to_ncop(cleaned_maf):
    """Convert MAF to nCop format"""
    print("\n" + "="*60)
    print("Step 3: Converting MAF to nCop Format (gene x patient)")
    print("="*60)
    
    # Select gene and patient columns
    ncop = cleaned_maf[['Tumor_Sample_Barcode', 'Hugo_Symbol']].drop_duplicates()
    
    print(f"Created gene-patient pairs: {len(ncop)}")
    print(f"  Unique genes: {ncop['Hugo_Symbol'].nunique()}")
    print(f"  Unique patients: {ncop['Tumor_Sample_Barcode'].nunique()}")
    
    # Save nCop format
    output_path = Path('data/mafs/ncop/sample_study_ncop.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ncop.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"Saved to: {output_path}")
    
    return ncop

def generate_gene_list(ncop_df):
    """Generate gene list for Endeavour"""
    print("\n" + "="*60)
    print("Step 4: Generating Gene List for Endeavour Analysis")
    print("="*60)
    
    genes = sorted(ncop_df['Hugo_Symbol'].unique())
    
    print(f"Extracted {len(genes)} unique genes")
    print(f"Sample genes: {genes[:5]}")
    
    # Save gene list
    output_path = Path('data/endeavour/MAF_lists/sample_study_genes.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for gene in genes:
            f.write(f"{gene}\n")
    
    print(f"Saved to: {output_path}")
    return genes

def create_sample_network(genes):
    """Create a sample gene network"""
    print("\n" + "="*60)
    print("Step 5: Creating Sample Gene Network")
    print("="*60)
    
    edges = []
    np.random.seed(42)
    
    # Create random interactions
    for i, gene1 in enumerate(genes):
        # Each gene connects to 2-5 other genes
        n_connections = np.random.randint(2, 6)
        partners = np.random.choice([g for g in genes if g != gene1], size=min(n_connections, len(genes)-1), replace=False)
        
        for gene2 in partners:
            confidence = np.random.uniform(0.4, 0.99)
            edges.append([gene1, gene2, confidence])
    
    network_df = pd.DataFrame(edges, columns=['gene1', 'gene2', 'confidence'])
    
    # Save network
    output_path = Path('data/networks/sample_network.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network_df.to_csv(output_path, sep='\t', index=False, header=False)
    
    print(f"Created network with {len(network_df)} edges")
    print(f"  Average degree: {len(network_df)*2/len(genes):.1f}")
    print(f"Saved to: {output_path}")
    
    return network_df

def simulate_endeavour_results(genes):
    """Simulate Endeavour ranking results"""
    print("\n" + "="*60)
    print("Step 6: Simulating Endeavour Ranking Results")
    print("="*60)
    
    # Create rankings (known drivers get better ranks)
    known_drivers = ['TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF']
    
    rankings = []
    for i, gene in enumerate(genes):
        if gene in known_drivers:
            rank = np.random.randint(1, 5)  # Top ranks for known drivers
        else:
            rank = np.random.randint(5, len(genes)+1)
        
        rankings.append({
            'Gene': gene,
            'Rank': rank,
            'Score': 1.0 / (rank + 1)
        })
    
    endeavour_df = pd.DataFrame(rankings).sort_values('Rank')
    
    # Save results
    output_path = Path('data/endeavour/results/sample_study_rankings.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    endeavour_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Created rankings for {len(endeavour_df)} genes")
    print(f"Top 5 ranked genes:")
    for _, row in endeavour_df.head().iterrows():
        print(f"  {row['Gene']}: Rank {row['Rank']}, Score {row['Score']:.3f}")
    print(f"Saved to: {output_path}")
    
    return endeavour_df

def convert_to_weights(endeavour_df):
    """Convert Endeavour rankings to nCop weights"""
    print("\n" + "="*60)
    print("Step 7: Converting Rankings to Network Weights")
    print("="*60)
    
    # Weight = 1 / (rank + 1)
    endeavour_df['weight'] = 1.0 / (endeavour_df['Rank'] + 1)
    
    # Normalize to [0, 1]
    endeavour_df['weight_normalized'] = (
        (endeavour_df['weight'] - endeavour_df['weight'].min()) /
        (endeavour_df['weight'].max() - endeavour_df['weight'].min())
    )
    
    print(f"Weight statistics:")
    print(f"  Min: {endeavour_df['weight_normalized'].min():.3f}")
    print(f"  Max: {endeavour_df['weight_normalized'].max():.3f}")
    print(f"  Mean: {endeavour_df['weight_normalized'].mean():.3f}")
    
    # Save weights
    output_path = Path('data/endeavour/ncop_weights/sample_study_weights.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    endeavour_df[['Gene', 'weight_normalized']].to_csv(output_path, sep='\t', index=False, header=False)
    
    print(f"Saved to: {output_path}")
    return endeavour_df

def test_gnn_modules(genes):
    """Test GNN embedding and clustering modules"""
    print("\n" + "="*60)
    print("Step 8: Testing GNN Modules (Embeddings & Clustering)")
    print("="*60)
    
    try:
        from gnn.embedding import Node2VecEmbedding
        from gnn.clustering import GraphClustering
        import networkx as nx
        
        # Load network
        network_path = Path('data/networks/sample_network.txt')
        edges = pd.read_csv(network_path, sep='\t', header=None, names=['source', 'target', 'weight'])
        
        # Create graph
        G = nx.Graph()
        for _, row in edges.iterrows():
            G.add_edge(row['source'], row['target'], weight=row['weight'])
        
        print(f"\nNetwork loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Generate embeddings
        print("\nGenerating Node2Vec embeddings...")
        embedder = Node2VecEmbedding(dimensions=8, walk_length=10, num_walks=20, workers=1)
        embedder.fit(G)
        embeddings_dict = embedder.get_embeddings()
        
        # Convert to numpy array
        nodes = sorted(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[node] for node in nodes])
        
        print(f"Embeddings created: {embeddings.shape[0]} genes, {embeddings.shape[1]} dimensions")
        
        # Save embeddings
        output_path = Path('data/networks/embeddings.txt')
        embedder.save_embeddings_txt(output_path)
        print(f"Saved to: {output_path}")
        
        # Clustering
        print("\nPerforming K-means clustering...")
        clusterer = GraphClustering(n_clusters=3, method='kmeans')
        labels = clusterer.fit_predict(embeddings)
        
        print(f"Genes clustered into {len(set(labels))} groups")
        for cluster_id in set(labels):
            count = sum(labels == cluster_id)
            print(f"  Cluster {cluster_id}: {count} genes")
        
        return embeddings, labels
        
    except Exception as e:
        print(f"Error in GNN modules: {e}")
        return None, None

def create_known_drivers():
    """Create sample known driver gene list"""
    print("\n" + "="*60)
    print("Step 9: Creating Known Driver Reference List")
    print("="*60)
    
    known_drivers = ['TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'NRAS', 'APC']
    
    output_path = Path('data/evaluation/reference/known_drivers.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for gene in known_drivers:
            f.write(f"{gene}\n")
    
    print(f"Created reference list with {len(known_drivers)} known drivers")
    print(f"Drivers: {', '.join(known_drivers)}")
    print(f"Saved to: {output_path}")
    
    return known_drivers

def evaluate_predictions(endeavour_df, known_drivers):
    """Evaluate prediction performance"""
    print("\n" + "="*60)
    print("Step 10: Evaluating Predictions")
    print("="*60)
    
    # Calculate precision at K
    k_values = [5, 10, 20]
    
    for k in k_values:
        top_k = set(endeavour_df.head(k)['Gene'].values)
        true_positives = len(top_k.intersection(set(known_drivers)))
        precision = true_positives / k
        
        print(f"\nPrecision@{k}: {precision:.3f}")
        print(f"  True positives: {true_positives}/{k}")
        print(f"  Genes found: {sorted(top_k.intersection(set(known_drivers)))}")

def main():
    """Run the complete demonstration workflow"""
    print("\n" + "#"*60)
    print("# Cancer Driver Network Analysis - Demo Workflow")
    print("#"*60)
    
    # Step 1-4: MAF processing
    maf_df = create_sample_maf()
    cleaned_maf = clean_maf(maf_df)
    ncop_df = maf_to_ncop(cleaned_maf)
    genes = generate_gene_list(ncop_df)
    
    # Step 5: Network creation
    network_df = create_sample_network(genes)
    
    # Step 6-7: Endeavour simulation
    endeavour_df = simulate_endeavour_results(genes)
    weights_df = convert_to_weights(endeavour_df)
    
    # Step 8: GNN modules
    embeddings, labels = test_gnn_modules(genes)
    
    # Step 9-10: Evaluation
    known_drivers = create_known_drivers()
    evaluate_predictions(endeavour_df, known_drivers)
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE!")
    print("="*60)
    print("\nAll sample data created in:")
    print("  - data/mafs/")
    print("  - data/networks/")
    print("  - data/endeavour/")
    print("  - data/evaluation/")
    print("\nYou can now explore the Jupyter notebooks to see")
    print("detailed implementations of each step.")

if __name__ == "__main__":
    main()
