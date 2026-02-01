#!/bin/bash
# =============================================================================
# Epsilon Cloud Setup Script for JarvisLabs
# =============================================================================
# This script automates the complete setup of Epsilon on a cloud GPU instance.
# Run this script after SSH-ing into your JarvisLabs instance.
#
# Usage: bash setup_cloud.sh
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║           EPSILON: Autonomous Research Engine                     ║"
echo "║                   Cloud Setup Script                              ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Configuration
REPO_URL="https://github.com/jhawaritvik/Epsilon.git"
BRANCH="main"
INSTALL_DIR="${HOME}/Epsilon"

# -----------------------------------------------------------------------------
# Step 1: Clone Repository
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Cloning Epsilon repository (branch: ${BRANCH})...${NC}"

if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}  Directory exists. Pulling latest changes...${NC}"
    cd "$INSTALL_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo -e "${GREEN}  ✓ Repository ready${NC}"

# -----------------------------------------------------------------------------
# Step 2: Set up Python Virtual Environment
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/6] Setting up Python virtual environment...${NC}"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}  ✓ Virtual environment already exists${NC}"
fi

source .venv/bin/activate
echo -e "${GREEN}  ✓ Virtual environment activated${NC}"

# -----------------------------------------------------------------------------
# Step 3: Install Dependencies
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/6] Installing dependencies...${NC}"

pip install --upgrade pip -q

# Check for GPU and install appropriate PyTorch
if command -v nvidia-smi &> /dev/null; then
    echo -e "${CYAN}  GPU detected - installing PyTorch with CUDA support...${NC}"
    # Install PyTorch with CUDA (adjust version as needed for JarvisLabs)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
else
    echo -e "${CYAN}  No GPU detected - installing CPU-only PyTorch...${NC}"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
fi

# Install remaining requirements (excluding torch lines to avoid conflicts)
grep -v "^torch" requirements.txt | pip install -r /dev/stdin -q

echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# -----------------------------------------------------------------------------
# Step 4: Configure Environment Variables
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/6] Configuring environment variables...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo -e "${CYAN}  Please enter your API keys:${NC}"
    echo ""
    
    read -p "  OpenAI API Key (sk-...): " OPENAI_KEY
    read -p "  Tavily API Key (tvly-...): " TAVILY_KEY
    read -p "  Supabase URL (https://xxx.supabase.co): " SUPABASE_URL
    read -p "  Supabase Key (anon key): " SUPABASE_KEY
    
    # Write to .env file
    cat > .env << EOF
OPENAI_API_KEY=${OPENAI_KEY}
TAVILY_API_KEY=${TAVILY_KEY}
SUPABASE_URL=${SUPABASE_URL}
SUPABASE_KEY=${SUPABASE_KEY}
EOF
    
    echo -e "${GREEN}  ✓ Environment configured${NC}"
else
    echo -e "${GREEN}  ✓ .env file already exists${NC}"
fi

# -----------------------------------------------------------------------------
# Step 5: Verify GPU Access
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/6] Checking GPU availability...${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU check failed")
    echo -e "${GREEN}  ✓ GPU detected: ${GPU_INFO}${NC}"
else
    echo -e "${YELLOW}  ⚠ No GPU detected (running on CPU)${NC}"
fi

# Check CUDA availability in Python
python -c "import torch; print(f'  PyTorch CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo -e "${YELLOW}  ⚠ PyTorch not installed or CUDA not available${NC}"

# -----------------------------------------------------------------------------
# Step 6: Run Quick Validation
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[6/6] Running quick validation...${NC}"

# Test imports
python -c "
from controller import ResearchController
from research_agent import research_agent
from experiment_agent import experiment_design_agent
from execution_agent import code_execution_agent
from evaluation_agent import evaluation_agent
print('  ✓ All core modules imported successfully')
" || echo -e "${RED}  ✗ Import validation failed${NC}"

# -----------------------------------------------------------------------------
# Step 7: Optional Docker Setup
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[Optional] Docker Execution Setup${NC}"
echo ""

if command -v docker &> /dev/null; then
    echo -e "${CYAN}  Docker is available. Would you like to build the execution container?${NC}"
    read -p "  Build Docker image for isolated execution? (y/N): " BUILD_DOCKER
    
    if [[ "$BUILD_DOCKER" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}  Building Docker image (this may take a few minutes)...${NC}"
        docker build -f docker/Dockerfile.execution -t epsilon-executor:latest . && \
            echo -e "${GREEN}  ✓ Docker image built successfully${NC}" || \
            echo -e "${RED}  ✗ Docker build failed${NC}"
        
        # Update .env to enable Docker
        if ! grep -q "USE_DOCKER" .env; then
            echo "USE_DOCKER=true" >> .env
            echo -e "${GREEN}  ✓ Docker execution enabled in .env${NC}"
        fi
    else
        echo -e "${YELLOW}  Skipping Docker setup (you can run this later)${NC}"
    fi
else
    echo -e "${YELLOW}  Docker not installed. Experiments will run locally.${NC}"
    echo "  To enable Docker isolation later, install Docker and run:"
    echo "    docker build -f docker/Dockerfile.execution -t epsilon-executor:latest ."
fi

# -----------------------------------------------------------------------------
# Complete!
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    SETUP COMPLETE!                                ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}To run Epsilon:${NC}"
echo ""
echo "  cd ${INSTALL_DIR}"
echo "  source .venv/bin/activate"
echo "  python main.py --goal \"Your research question here\""
echo ""
echo -e "${CYAN}To run tests:${NC}"
echo ""
echo "  pytest tests/ -v"
echo ""
echo -e "${YELLOW}Branch: ${BRANCH}${NC}"
echo ""
