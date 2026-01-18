# Cancer Driver Network Analysis

A comprehensive pipeline for identifying cancer driver genes through network-based prioritization and graph neural network analysis.

## Project Overview

This project implements an end-to-end analysis pipeline that integrates:
- Somatic mutation data (MAF) from cBioPortal
- Functional interaction networks (HPRD / STRING)
- Gene prioritization (Endeavour-style enrichment)
- Network-aware mutation analysis (nCop-style weighting)
- Graph embeddings and clustering for validation

### Pipeline Overview
1. MAF preprocessing and cleaning
2. Conversion to network-compatible formats
3. Gene list generation from mutation data and networks
4. Gene prioritization and network weighting
5. Network-based driver identification
6. Evaluation using known driver gene databases
7. Graph embedding and cluster-based validation

## Project Structure

```
cancer-driver-network-analysis/
│
├── data/
│   ├── mafs/              # MAF (Mutation Annotation Format) files
│   │   ├── raw/           # Raw MAF files
│   │   ├── clean/         # Cleaned MAF files
│   │   └── ncop/          # NCOP-formatted data
│   ├── networks/          # Network data files
│   ├── endeavour/         # Endeavour analysis results
│   └── evaluation/        # Evaluation results
│
├── notebooks/
│   ├── 01_maf_cleaning.ipynb              # MAF file cleaning and preprocessing
│   ├── 02_maf_to_ncop.ipynb              # Convert MAF to NCOP format
│   ├── 03_gene_list_generation.ipynb     # Generate gene lists
│   ├── 04_endeavour_to_ncop_weights.ipynb # Convert Endeavour to NCOP weights
│   ├── 05_ncop_script_generator.ipynb    # Generate NCOP scripts
│   └── 06_evaluation.ipynb                # Evaluate results
│
├── gnn/
│   ├── embedding.py            # Graph embedding implementations
│   ├── clustering.py           # Clustering algorithms
│   └── correlation_analysis.py # Correlation analysis tools
│
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- 4GB+ RAM
- MAF files from cBioPortal or TCGA

## Complete Pipeline Workflow

### Step 1: Data Preparation (Notebooks 01-03)

#### 01_maf_cleaning.ipynb
**Goal**: Clean and filter MAF files

1. Download MAF files from [cBioPortal](https://www.cbioportal.org/)
2. Place in `data/mafs/raw/`
3. Run notebook to:
   - Remove silent mutations
   - Keep only protein-altering variants
   - Standardize gene names
   - Output cleaned files to `data/mafs/clean/`

**Example**:
```python
# Filters for: Missense, Nonsense, Frame_Shift, Splice_Site mutations
# Input: data/mafs/raw/*.maf
# Output: data/mafs/clean/*.maf
```

#### 02_maf_to_ncop.ipynb
**Goal**: Convert to nCop format (gene × patient)

```python
# Input: data/mafs/clean/*.maf
# Output: data/mafs/ncop/*.maf
```

#### 03_gene_list_generation.ipynb
**Goal**: Generate gene lists for Endeavour

```python
# Input: data/mafs/ncop/*.maf
# Output: data/endeavour/MAF_lists/*_genes.txt
```

### Step 2: Functional Network Analysis (External)

1. **Download Interaction Networks**:
   - [STRING](https://string-db.org/): Protein-protein interactions
   - [BioGRID](https://thebiogrid.org/): Curated interactions
   - [HPRD](http://www.hprd.org/): Human protein reference

2. **Save to**: `data/networks/`

3. **Extract network gene lists** (use notebook 03 logic)

### Step 3: Endeavour Prioritization (External Tool)
## Data Formats

### MAF (Mutation Annotation Format)
Standard format for somatic mutations. Required columns:
- `Hugo_Symbol`: Gene name
- `Variant_Classification`: Mutation type
- `Tumor_Sample_Barcode`: Patient ID

### nCop Format
Simplified format for network analysis:
```
Tumor_Sample_Barcode    Hugo_Symbol
TCGA-01-0001           TP53
TCGA-01-0001           KRAS
...
```

### Endeavour Input
Plain text, one gene per line:
```
TP53
KRAS
EGFR
...
```

### Network Format
Tab-separated edge list:
```
gene1    gene2    weight
TP53     MDM2     0.95
KRAS     BRAF     0.87
...
```

## Expected Results

### Typical Performance (Top-50)
- **Precision**: 40-60%
- **Recall**: 5-15%
- **F1-Score**: 9-24%

Performance depends on:
- Quality of input MAF data
- Network coverage
- Cancer type specificity

### Example Output
```
Top 20 Predicted Drivers:
1. TP53        [Known Driver]
2. KRAS        [Known Driver]
3. EGFR        [Known Driver]
4. PIK3CA      [Known Driver]
5. NOVEL_GENE  [Novel Candidate]
...
```

## Pipeline Diagram

```
MAF Files (cBioPortal)
         ↓
    [01_cleaning]
         ↓
   Cleaned MAFs
         ↓
  [02_to_ncop]
         ↓
    nCop Format → [03_gene_lists] → Gene Lists
                                         ↓
                                   [Endeavour] ← Networks
                                         ↓
                                    Rankings
                                         ↓
                              [04_to_weights]
                                         ↓
                              [05_propagation]
                                         ↓
                                    Results
                                         ↓
                                [06_evaluation]
                                         ↓
                            Performance Metrics
```

## Troubleshooting

### Common Issues

**No MAF files found**
```bash
# Ensure files are in correct location
ls data/mafs/raw/
# Should show: *.maf files
```

**Column name errors**
- Notebooks automatically handle common column name variations
- Manually check column names: `df.columns`

**Memory issues with large files**
- Use `low_memory=False` in pandas (already set)
- Process files one at a time
- Consider downsampling for testing

**Network convergence issues**
- Increase `max_iterations` in config
- Adjust `restart_probability` (default: 0.15)
- Check network connectivity

## Resources

### Data Sources
- **cBioPortal**: https://www.cbioportal.org/
- **TCGA**: https://portal.gdc.cancer.gov/
- **COSMIC**: https://cancer.sanger.ac.uk/census
- **IntOGen**: https://www.intogen.org/

### Networks
- **HPRD**: http://www.hprd.org/
- **STRING**: https://string-db.org/
- **BioGRID**: https://thebiogrid.org/

### Tools
- **Endeavour**: https://endeavour.esat.kuleuven.be/

### References
1. Bailey et al. (2018). "Comprehensive Characterization of Cancer Driver Genes" Cell
2. Vanunu et al. (2010). "Associating Genes and Protein Complexes with Disease via Network Propagation" PLoS Comp Bio
3. Grover & Leskovec (2016). "node2vec: Scalable Feature Learning for Networks" KDD
4. TCGA Research Network. "The Cancer Genome Atlas Program"

## Reproducibility

All steps can be reproduced by:
- Running notebooks in order (01-06)
- Using the provided Snakemake workflow for automated execution
- All random seeds are set for deterministic results

## Contributing

Contributions welcome! Areas for improvement:
- Additional embedding methods (GraphSAGE, GAT)
- Multi-omics integration
- Drug target prediction
- Pathway enrichment analysis

## License

MIT License - feel free to use for research and academic purposes.

## Acknowledgments

This pipeline integrates multiple open-source tools and databases. Please cite appropriate sources when using this work.

## Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: January 2026
**Version**: 1.0.0
# Output: data/endeavour/ncop_weights/*.txt
```

Generates:
- Normalized weights [0, 1]
- Weight distribution visualizations

### Step 5: Network Propagation (Notebook 05)

#### 05_ncop_script_generator.ipynb
**Goal**: Generate network propagation scripts

Creates Python scripts that:
- Load interaction networks
- Apply gene weights
- Perform random walk with restart
- Prioritize driver genes

**Run generated script**:
```bash
cd data/ncop_scripts
python run_ncop_analysis.py
```

### Step 6: Evaluation (Notebook 06)

#### 06_evaluation.ipynb
**Goal**: Validate against known drivers

Compares predictions with:
- **COSMIC Cancer Gene Census**
- **IntOGen**
- **Bailey et al. 2018**

**Metrics**:
- Precision @ K
- Recall @ K
- F1-score @ K
- ROC curves

**Outputs**:
- `data/evaluation/evaluation_metrics.png`
- `data/evaluation/evaluation_summary.csv`

### Step 7: Advanced GNN Analysis (Optional)

#### GNN Module Usage

```python
from gnn.embedding import GraphEmbedding
from gnn.clustering import GraphClustering
from gnn.correlation_analysis import CorrelationAnalysis

# Load network
import networkx as nx
G = nx.read_edgelist('data/networks/functional_network.txt')

# Generate embeddings
embedder = GraphEmbedding(method='node2vec', dimensions=128)
embeddings = embedder.fit_transform(G)

# Cluster genes
clusterer = GraphClustering(n_clusters=10, method='kmeans')
labels = clusterer.fit_predict(embeddings)

# Analyze correlations
correlator = CorrelationAnalysis(method='pearson')
corr_matrix = correlator.compute_correlation_matrix(embeddings)
```

## Key Features

### Data Processing
- Automated MAF file cleaning
- Multiple format conversions
- Flexible column name handling
- Batch processing support

### Network Analysis
- Random walk with restart propagation
- Weighted gene prioritization
- Multiple network integration
- Convergence monitoring

### Machine Learning
- Node2Vec embeddings
- DeepWalk embeddings
- Graph clustering (K-means, Spectral)
- Correlation analysis

### Evaluation
- Precision/Recall/F1 metrics
- Top-K evaluation
- Visualization suite
- Comprehensive reporting
jupyter notebook notebooks/01_maf_cleaning.ipynb
```

## Usage

### Data Preparation

1. Place raw MAF files in `data/mafs/raw/`
2. Run notebooks in order:
   - `01_maf_cleaning.ipynb` - Clean and preprocess MAF files
   - `02_maf_to_ncop.ipynb` - Convert to NCOP format
   - `03_gene_list_generation.ipynb` - Generate gene lists for analysis

### Network Analysis

3. Continue with network analysis notebooks:
   - `04_endeavour_to_ncop_weights.ipynb` - Process Endeavour data
   - `05_ncop_script_generator.ipynb` - Generate NCOP scripts
   - `06_evaluation.ipynb` - Evaluate and visualize results

### GNN Analysis

The `gnn/` module provides tools for:
- **Graph Embedding**: Create node embeddings from network structures
- **Clustering**: Identify communities and modules in cancer driver networks
- **Correlation Analysis**: Analyze relationships between genes and features

## Data Format

### MAF Files
Standard Mutation Annotation Format files containing somatic mutations.

### NCOP Format
Network-based format used for cancer pathway analysis.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
