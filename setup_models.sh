#!/bin/bash

# Model Setup Script for Slime
# This script downloads models and datasets, then converts them to Megatron format

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}     Slime Model Setup and Conversion Script${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Check if we're in the slime directory
if [ ! -f "train.py" ] || [ ! -d "slime" ]; then
    echo -e "${RED}Error: Please run this script from the slime root directory${NC}"
    exit 1
fi

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install -U huggingface_hub
fi

# Configuration
MODELS_DIR="${MODELS_DIR:-/root}"
SLIME_ROOT="$(pwd)"

echo -e "${GREEN}Models will be saved to: ${MODELS_DIR}${NC}"
echo ""

# Function to download model
download_model() {
    local model_name=$1
    local model_path=$2
    local repo_id=$3
    
    if [ -d "$model_path" ]; then
        echo -e "${YELLOW}Model already exists at $model_path, skipping download${NC}"
        return 0
    fi
    
    echo -e "${BLUE}Downloading $model_name...${NC}"
    huggingface-cli download "$repo_id" --local-dir "$model_path"
    echo -e "${GREEN}✓ Downloaded $model_name${NC}"
}

# Function to download dataset
download_dataset() {
    local dataset_name=$1
    local dataset_path=$2
    local repo_id=$3
    
    if [ -d "$dataset_path" ]; then
        echo -e "${YELLOW}Dataset already exists at $dataset_path, skipping download${NC}"
        return 0
    fi
    
    echo -e "${BLUE}Downloading $dataset_name...${NC}"
    huggingface-cli download --repo-type dataset "$repo_id" --local-dir "$dataset_path"
    echo -e "${GREEN}✓ Downloaded $dataset_name${NC}"
}

# Function to convert model
convert_model() {
    local model_name=$1
    local hf_path=$2
    local torch_dist_path=$3
    local model_config=$4
    
    if [ -d "$torch_dist_path" ]; then
        echo -e "${YELLOW}Converted model already exists at $torch_dist_path, skipping conversion${NC}"
        return 0
    fi
    
    echo -e "${BLUE}Converting $model_name to Megatron format...${NC}"
    echo -e "Source: $hf_path"
    echo -e "Target: $torch_dist_path"
    
    # Load model configuration
    source "$SLIME_ROOT/scripts/models/$model_config"
    
    # Run conversion
    PYTHONPATH=/root/Megatron-LM python "$SLIME_ROOT/tools/convert_hf_to_torch_dist.py" \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "$hf_path" \
        --save "$torch_dist_path"
    
    echo -e "${GREEN}✓ Converted $model_name${NC}"
}

# ============================================================
# Step 1: Download Models
# ============================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: Downloading Models${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Qwen3-0.6B (the actual Qwen3 model)
download_model "Qwen3-0.6B" \
    "$MODELS_DIR/Qwen3-0.6B" \
    "Qwen/Qwen3-0.6B"

# GLM-4-9B (using GLM-Z1-9B as mentioned in docs)
download_model "GLM-4-9B" \
    "$MODELS_DIR/GLM-Z1-9B-0414" \
    "zai-org/GLM-Z1-9B-0414"

echo ""

# ============================================================
# Step 2: Download Datasets
# ============================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Downloading Datasets${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Training dataset
download_dataset "dapo-math-17k" \
    "$MODELS_DIR/dapo-math-17k" \
    "zhuzilin/dapo-math-17k"

# Evaluation dataset
download_dataset "aime-2024" \
    "$MODELS_DIR/aime-2024" \
    "zhuzilin/aime-2024"

echo ""

# ============================================================
# Step 3: Convert Models to Megatron Format
# ============================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Converting Models to Megatron Format${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Convert Qwen3-0.6B
convert_model "Qwen3-0.6B" \
    "$MODELS_DIR/Qwen3-0.6B" \
    "$MODELS_DIR/Qwen3-0.6B_torch_dist" \
    "qwen3-0.6B.sh"

# Convert GLM-4-9B
convert_model "GLM-4-9B" \
    "$MODELS_DIR/GLM-Z1-9B-0414" \
    "$MODELS_DIR/GLM-Z1-9B-0414_torch_dist" \
    "glm4-9B.sh"

echo ""

# ============================================================
# Summary
# ============================================================

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}           Setup Complete! 🎉${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Models and datasets are ready:"
echo ""
echo -e "${BLUE}Qwen3-0.6B:${NC}"
echo -e "  HF checkpoint: $MODELS_DIR/Qwen3-0.6B"
echo -e "  Torch dist:    $MODELS_DIR/Qwen3-0.6B_torch_dist"
echo ""
echo -e "${BLUE}GLM-4-9B:${NC}"
echo -e "  HF checkpoint: $MODELS_DIR/GLM-Z1-9B-0414"
echo -e "  Torch dist:    $MODELS_DIR/GLM-Z1-9B-0414_torch_dist"
echo ""
echo -e "${BLUE}Datasets:${NC}"
echo -e "  Training:   $MODELS_DIR/dapo-math-17k/"
echo -e "  Evaluation: $MODELS_DIR/aime-2024/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Review and modify scripts/run-qwen3-06B-copy.sh if needed"
echo -e "2. Run: ${GREEN}bash scripts/run-qwen3-06B-copy.sh${NC}"
echo ""

