# Docker Build and Run Instructions

## Build the Docker Image

```bash
docker build -t cancer-driver-network-analysis:latest .
```

## Run Demo Workflow

```bash
docker run --rm -v $(pwd)/data:/app/data cancer-driver-network-analysis:latest
```

## Run with Docker Compose

### Run Demo Workflow
```bash
docker-compose up cancer-analysis
```

### Start Jupyter Lab
```bash
docker-compose up jupyter
```

Then open http://localhost:8888 in your browser.

## Run Interactive Shell

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/notebooks:/app/notebooks \
  cancer-driver-network-analysis:latest \
  /bin/bash
```

## Run Specific Analysis

### Generate Embeddings
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  cancer-driver-network-analysis:latest \
  python gnn/embedding.py data/networks/sample_network.txt data/networks/embeddings.txt
```

### Run Clustering
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  cancer-driver-network-analysis:latest \
  python gnn/clustering.py data/networks/embeddings.txt data/networks/clusters.csv 3
```

## Mount Your Own Data

```bash
docker run --rm \
  -v /path/to/your/maf/files:/app/data/mafs/raw \
  -v /path/to/your/networks:/app/data/networks \
  -v /path/to/output:/app/results \
  cancer-driver-network-analysis:latest \
  python demo_workflow.py
```

## Cleanup

```bash
docker-compose down
docker rmi cancer-driver-network-analysis:latest
```
