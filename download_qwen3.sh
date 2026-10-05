#!/bin/bash

# Quick script to download the correct Qwen3-0.6B model and verify config

set -e

echo "========================================"
echo "Downloading Qwen3-0.6B Model"
echo "========================================"
echo ""

# Install huggingface_hub if not already installed
pip install -q -U huggingface_hub

# Download the actual Qwen3-0.6B model
echo "Downloading Qwen/Qwen3-0.6B to /root/Qwen3-0.6B..."
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /root/Qwen3-0.6B

echo ""
echo "========================================"
echo "Model Configuration Check"
echo "========================================"
echo ""

# Check the actual configuration
python3 << 'EOF'
from transformers import AutoConfig
import json

try:
    config = AutoConfig.from_pretrained('/root/Qwen3-0.6B', trust_remote_code=True)
    
    print("✓ Qwen3-0.6B Model Configuration:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  rms_norm_eps: {getattr(config, 'rms_norm_eps', 'N/A')}")
    print(f"  rope_theta: {getattr(config, 'rope_theta', 'N/A')}")
    print(f"  max_position_embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
    print("")
    print("Model successfully downloaded and verified!")
    
except Exception as e:
    print(f"Error: {e}")
    exit(1)
EOF

echo ""
echo "========================================"
echo "Next Steps"
echo "========================================"
echo ""
echo "1. Update scripts/models/qwen3-0.6B.sh with the correct configuration"
echo "2. Convert model to Megatron format:"
echo "   source scripts/models/qwen3-0.6B.sh"
echo "   PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \\"
echo "       \${MODEL_ARGS[@]} \\"
echo "       --hf-checkpoint /root/Qwen3-0.6B \\"
echo "       --save /root/Qwen3-0.6B_torch_dist"
echo ""
echo "3. Run the training script:"
echo "   bash scripts/run-qwen3-06B-copy.sh"
echo ""


