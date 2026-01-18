"""
Snakemake Workflow for Cancer Driver Network Analysis

This workflow automates the entire analysis pipeline from MAF files
to final evaluation and graph embeddings.

Run with: snakemake --cores 4
"""

# Configuration
NETWORK_FILE = "data/networks/network.edgelist"
EMBEDDINGS_FILE = "data/networks/embeddings.txt"
CLUSTERS_FILE = "data/networks/gene_clusters.csv"
CORRELATIONS_FILE = "data/evaluation/top_correlations.csv"

# Main rule - defines all final outputs
rule all:
    input:
        EMBEDDINGS_FILE,
        CLUSTERS_FILE,
        CORRELATIONS_FILE

# Rule 1: Generate network embeddings using Node2Vec
rule embed_network:
    input:
        NETWORK_FILE
    output:
        EMBEDDINGS_FILE
    log:
        "logs/embed_network.log"
    shell:
        """
        python gnn/embedding.py {input} {output} 2> {log}
        """

# Rule 2: Cluster genes based on embeddings
rule cluster_genes:
    input:
        EMBEDDINGS_FILE
    output:
        CLUSTERS_FILE
    log:
        "logs/cluster_genes.log"
    shell:
        """
        python gnn/clustering.py {input} {output} 2> {log}
        """

# Rule 3: Correlate with known drivers
rule correlate_drivers:
    input:
        embeddings=EMBEDDINGS_FILE,
        drivers="data/evaluation/reference/known_cancer_drivers.txt"
    output:
        CORRELATIONS_FILE
    log:
        "logs/correlate_drivers.log"
    shell:
        """
        python gnn/correlation_analysis.py {input.embeddings} {input.drivers} {output} 2> {log}
        """

# Optional: Clean intermediate files
rule clean:
    shell:
        """
        rm -rf data/networks/embeddings.txt
        rm -rf data/networks/gene_clusters.csv
        rm -rf data/evaluation/top_correlations.csv
        rm -rf logs/*.log
        """

# Optional: Clean all generated data
rule clean_all:
    shell:
        """
        rm -rf data/mafs/clean/*
        rm -rf data/mafs/ncop/*
        rm -rf data/endeavour/MAF_lists/*
        rm -rf data/endeavour/ncop_weights/*
        rm -rf data/networks/embeddings.txt
        rm -rf data/networks/gene_clusters.csv
        rm -rf data/evaluation/*.csv
        rm -rf data/evaluation/*.png
        rm -rf logs/*.log
        """
