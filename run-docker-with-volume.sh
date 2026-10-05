#!/bin/bash

# Docker Container Launcher with Volume Mount
# Based on the quick_start.md guide but with persistent volume mounting

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}     Slime Docker Container with Volume Mount${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Configuration
IMAGE_NAME="${SLIME_IMAGE:-slimerl/slime:latest}"
CONTAINER_NAME="${SLIME_CONTAINER_NAME:-slime-dev}-mjacob1002-2.0"
SLIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Image: ${BLUE}$IMAGE_NAME${NC}"
echo -e "  Container: ${BLUE}$CONTAINER_NAME${NC}"
echo -e "  Host directory: ${BLUE}$SLIME_DIR${NC}"
echo -e "  Container directory: ${BLUE}/workspace/slime${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: GPU support not detected${NC}"
    echo -e "${YELLOW}Container will start without GPU access${NC}"
    GPU_FLAGS=""
else
    echo -e "${GREEN}✓ GPU support detected${NC}"
    GPU_FLAGS="--gpus all"
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Container '${CONTAINER_NAME}' already exists${NC}"
    echo ""
    echo "Choose an option:"
    echo "  1) Start and attach to existing container"
    echo "  2) Execute new bash session in existing container"
    echo "  3) Stop and remove existing container, then create new one"
    echo "  4) Exit"
    read -p "Enter choice [1-4]: " choice
    
    case $choice in
        1)
            echo -e "${BLUE}Starting existing container...${NC}"
            docker start "$CONTAINER_NAME"
            docker attach "$CONTAINER_NAME"
            exit 0
            ;;
        2)
            echo -e "${BLUE}Executing new bash session...${NC}"
            docker exec -it "$CONTAINER_NAME" /bin/bash
            exit 0
            ;;
        3)
            echo -e "${BLUE}Removing existing container...${NC}"
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME"
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
fi

# Check if image exists locally
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${YELLOW}Image $IMAGE_NAME not found locally${NC}"
    echo -e "${BLUE}Pulling image from Docker Hub...${NC}"
    docker pull "$IMAGE_NAME"
fi

echo ""
echo -e "${BLUE}Starting new container...${NC}"
echo -e "${GREEN}Volume mount:${NC} $SLIME_DIR → /workspace/slime"
echo ""
echo -e "${YELLOW}Note: Changes made inside /workspace/slime will persist on your host!${NC}"
echo ""

# Start container with volume mount
# Based on quick_start.md but with added volume mount
docker run -it \
    --name "$CONTAINER_NAME" \
    $GPU_FLAGS \
    --ipc=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=524288:524288 \
    --network host \
    -v "$SLIME_DIR:/workspace/slime" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY="${DISPLAY:-}" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}" \
    -e PYTHONUNBUFFERED=1 \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -w /workspace/slime \
    "$IMAGE_NAME" \
    /bin/bash

echo ""
echo -e "${GREEN}Container exited${NC}"
echo -e "To restart: ${BLUE}docker start -ai $CONTAINER_NAME${NC}"
echo -e "To remove: ${BLUE}docker rm $CONTAINER_NAME${NC}"

