#!/bin/bash

# Cancer Driver Network Analysis - Setup Script
# This script sets up the complete environment

echo "=========================================="
echo "Cancer Driver Network Analysis Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo " Python 3 not found! Please install Python 3.8+"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo " Failed to create virtual environment"
    exit 1
fi

echo " Virtual environment created"

# Activate virtual environment (for this script)
echo ""
echo "Installing packages..."

# Use venv pip directly (works across shells)
./venv/bin/pip install --upgrade pip
./venv/bin/pip install pandas numpy matplotlib seaborn scipy scikit-learn networkx jupyter pyarrow gensim

if [ $? -ne 0 ]; then
    echo " Package installation failed"
    exit 1
fi

echo " Packages installed"

# Save requirements
echo ""
echo "Saving requirements.txt..."
./venv/bin/pip freeze > requirements.txt
echo " Requirements saved"

# Create all necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/mafs/{raw,clean,ncop}
mkdir -p data/networks
mkdir -p data/endeavour/{MAF_lists,results,ncop_weights}
mkdir -p data/evaluation/reference
mkdir -p data/ncop_scripts

echo " Directory structure created"

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo ""
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Data directories
data/

# Virtual environment
venv/
env/
.venv

# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]
*.so

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Environment
.env
EOF
    echo " .gitignore created"
fi

# Print success message
echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download MAF files from cBioPortal:"
echo "   https://www.cbioportal.org/"
echo ""
echo "3. Place MAF files in:"
echo "   data/mafs/raw/"
echo ""
echo "4. Start Jupyter:"
echo "   jupyter notebook"
echo ""
echo "5. Open and run notebooks in order:"
echo "   - 01_maf_cleaning.ipynb"
echo "   - 02_maf_to_ncop.ipynb"
echo "   - 03_gene_list_generation.ipynb"
echo "   - 04_endeavour_to_ncop_weights.ipynb"
echo "   - 05_ncop_script_generator.ipynb"
echo "   - 06_evaluation.ipynb"
echo ""
echo "For more information, see README.md"
echo ""
