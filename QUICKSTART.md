# Cancer Driver Network Analysis - Quick Reference

## 🚀 Quick Start Commands

```bash
# Setup (one-time)
./setup.sh

# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook

# Or start specific notebook
jupyter notebook notebooks/01_maf_cleaning.ipynb
```

## 📊 Notebook Execution Order

| # | Notebook | Input | Output | Purpose |
|---|----------|-------|--------|---------|
| 1 | `01_maf_cleaning.ipynb` | `data/mafs/raw/*.maf` | `data/mafs/clean/*.maf` | Filter protein-altering mutations |
| 2 | `02_maf_to_ncop.ipynb` | `data/mafs/clean/*.maf` | `data/mafs/ncop/*.maf` | Convert to gene×patient format |
| 3 | `03_gene_list_generation.ipynb` | `data/mafs/ncop/*.maf` | `data/endeavour/MAF_lists/*.txt` | Extract unique gene lists |
| 4 | `04_endeavour_to_ncop_weights.ipynb` | `data/endeavour/results/*.txt` | `data/endeavour/ncop_weights/*.txt` | Convert rankings to weights |
| 5 | `05_ncop_script_generator.ipynb` | Multiple inputs | `data/ncop_scripts/*.py` | Generate propagation scripts |
| 6 | `06_evaluation.ipynb` | Result files | `data/evaluation/*` | Calculate metrics & visualizations |

## 📥 Required External Data

### 1. MAF Files
- **Source**: [cBioPortal](https://www.cbioportal.org/)
- **Location**: `data/mafs/raw/`
- **Format**: Tab-separated, contains mutation data

### 2. Interaction Networks
- **STRING**: https://string-db.org/
- **BioGRID**: https://thebiogrid.org/
- **HPRD**: http://www.hprd.org/
- **Location**: `data/networks/`
- **Format**: Tab-separated edge list

### 3. Known Drivers (for evaluation)
- **COSMIC**: https://cancer.sanger.ac.uk/census
- **IntOGen**: https://www.intogen.org/
- **Location**: `data/evaluation/reference/`
- **Format**: One gene per line

### 4. Endeavour Results
- **Tool**: https://endeavour.esat.kuleuven.be/
- **Input**: Gene lists from notebook 03
- **Location**: `data/endeavour/results/`
- **Format**: Tab-separated with gene and rank columns

## 🔧 Key Configuration Parameters

### MAF Cleaning (Notebook 01)
```python
protein_altering = [
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site"
]
```

### Network Propagation (Notebook 05)
```python
ncop_config = {
    'network_type': 'functional_interaction',
    'weight_threshold': 0.1,
    'max_iterations': 1000,
    'convergence_threshold': 1e-6,
    'restart_probability': 0.15,  # Alpha parameter
}
```

### Node2Vec Embedding (GNN Module)
```python
embedder = GraphEmbedding(
    method='node2vec',
    dimensions=128,
    walk_length=80,
    num_walks=10,
    p=1.0,  # Return parameter
    q=1.0   # In-out parameter
)
```

### Evaluation Thresholds (Notebook 06)
```python
top_k_list = [10, 20, 50, 100, 200]  # Evaluate precision@K
```

## 📈 Typical Workflow Timeline

1. **Data Preparation** (1-2 hours)
   - Download MAF files
   - Run notebooks 01-03
   - Generate gene lists

2. **Endeavour Analysis** (varies)
   - Upload gene lists
   - Run prioritization
   - Download results

3. **Network Analysis** (30 min - 2 hours)
   - Download networks
   - Run notebooks 04-05
   - Execute propagation

4. **Evaluation** (15-30 min)
   - Download reference datasets
   - Run notebook 06
   - Generate reports

**Total**: ~4-8 hours (excluding Endeavour processing)

## 🎯 Expected File Sizes

| Data Type | Size Range | Notes |
|-----------|------------|-------|
| Raw MAF | 10MB - 5GB | Depends on cohort size |
| Cleaned MAF | 1MB - 500MB | ~10-20% of raw |
| nCop format | 500KB - 100MB | Simplified format |
| Gene lists | 1KB - 50KB | Text files |
| Networks | 5MB - 500MB | Depends on database |
| Results | 100KB - 10MB | Prioritized genes |

## 🐛 Common Issues & Solutions

### Issue: No MAF files found
**Solution**: Ensure files have `.maf` extension and are in `data/mafs/raw/`

### Issue: Column not found error
**Solution**: Notebooks handle common variations. Check: `Hugo_Symbol`, `Gene`, `Variant_Classification`

### Issue: Out of memory
**Solution**: 
- Process files individually
- Use `low_memory=False` (already set)
- Increase system RAM or use subsample

### Issue: Endeavour results not loading
**Solution**: Check file format - should be tab-separated with `gene` and `rank` columns

### Issue: Network doesn't converge
**Solution**:
- Increase `max_iterations`
- Check network connectivity
- Adjust `restart_probability`

### Issue: Low evaluation scores
**Expected**: Initial precision @50 typically 40-60%
**Improve**:
- Use cancer-specific networks
- Filter by mutation frequency
- Integrate multiple data sources

## 📚 Python Package Versions (Core)

```
pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
networkx >= 2.6.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
jupyter >= 1.0.0
```

## 🔬 GNN Module Usage Examples

### Example 1: Generate Embeddings
```python
from gnn.embedding import GraphEmbedding
import networkx as nx

# Load network
G = nx.read_edgelist('data/networks/string_network.txt')

# Create embeddings
embedder = GraphEmbedding(method='node2vec', dimensions=128)
embeddings = embedder.fit_transform(G)

# Save embeddings
import pickle
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
```

### Example 2: Cluster Genes
```python
from gnn.clustering import GraphClustering
import numpy as np

# Convert embeddings to matrix
genes = list(embeddings.keys())
X = np.array([embeddings[g] for g in genes])

# Cluster
clusterer = GraphClustering(n_clusters=10, method='kmeans')
labels = clusterer.fit_predict(X)

# Analyze clusters
for i in range(10):
    cluster_genes = [genes[j] for j in range(len(genes)) if labels[j] == i]
    print(f"Cluster {i}: {len(cluster_genes)} genes")
```

### Example 3: Correlation Analysis
```python
from gnn.correlation_analysis import CorrelationAnalysis

# Analyze correlations
analyzer = CorrelationAnalysis(method='pearson')
corr_matrix = analyzer.compute_correlation_matrix(X)

# Find significant correlations
sig_pairs = analyzer.identify_significant_correlations(
    corr_matrix, 
    threshold=0.7
)

print(f"Found {len(sig_pairs)} significant correlations")
```

## 📊 Interpreting Results

### Precision @K
- **> 60%**: Excellent
- **40-60%**: Good
- **20-40%**: Fair
- **< 20%**: Poor (check data quality)

### Recall @K
- **> 15%**: Excellent (for K=50-100)
- **10-15%**: Good
- **5-10%**: Fair
- **< 5%**: Limited coverage

### F1-Score @K
- Harmonic mean of precision and recall
- Best overall metric for balanced assessment
- **> 20%**: Good performance

## 🎓 Citation

If you use this pipeline, please cite:

```bibtex
@software{cancer_driver_network_analysis,
  title={Cancer Driver Network Analysis Pipeline},
  year={2026},
  version={1.0.0},
  url={https://github.com/yourusername/cancer-driver-network-analysis}
}
```

Also cite relevant tools and databases used in your analysis.

## 📞 Support

- **Issues**: Open a GitHub issue
- **Documentation**: See `README.md`
- **Examples**: Check notebook comments

---

**Version**: 1.0.0 | **Last Updated**: January 2026
