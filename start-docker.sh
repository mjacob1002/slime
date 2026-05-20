#!/bin/bash

# Slime Docker Container Startup Script
# This script starts a Docker container with Slime mounted as a volume

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="${SLIME_IMAGE:-slimerl/slime:latest}"
CONTAINER_NAME="slime-dev"
SLIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}        Slime Docker Container Startup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-docker runtime not found or not working${NC}"
    echo -e "${YELLOW}Container will start without GPU support${NC}"
    GPU_FLAGS=""
else
    echo -e "${GREEN}✓ GPU support detected${NC}"
    GPU_FLAGS="--gpus all"
fi

# Check if image exists locally
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${YELLOW}Image $IMAGE_NAME not found locally${NC}"
    echo -e "Would you like to:"
    echo "  1) Pull the image from Docker Hub"
    echo "  2) Build the image locally"
    echo "  3) Exit"
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        1)
            echo -e "${BLUE}Pulling $IMAGE_NAME...${NC}"
            docker pull "$IMAGE_NAME"
            ;;
        2)
            echo -e "${BLUE}Building image locally...${NC}"
            cd "$SLIME_DIR/docker"
            docker build \
                --build-arg SGLANG_VERSION=latest \
                --build-arg MEGATRON_COMMIT=core_v0.14.0 \
                -t "$IMAGE_NAME" \
                -f Dockerfile \
                .
            cd "$SLIME_DIR"
            ;;
        3)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Container ${CONTAINER_NAME} already exists${NC}"
    
    # Check if it's running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}Container is already running${NC}"
        echo "Do you want to:"
        echo "  1) Attach to the running container"
        echo "  2) Execute a new bash session"
        echo "  3) Restart the container"
        echo "  4) Stop and remove the container"
        echo "  5) Exit"
        read -p "Enter choice [1-5]: " choice
        
        case $choice in
            1)
                echo -e "${BLUE}Attaching to container...${NC}"
                docker attach "$CONTAINER_NAME"
                ;;
            2)
                echo -e "${BLUE}Starting new bash session...${NC}"
                docker exec -it "$CONTAINER_NAME" /bin/bash
                ;;
            3)
                echo -e "${BLUE}Restarting container...${NC}"
                docker restart "$CONTAINER_NAME"
                docker exec -it "$CONTAINER_NAME" /bin/bash
                ;;
            4)
                echo -e "${BLUE}Stopping and removing container...${NC}"
                docker stop "$CONTAINER_NAME"
                docker rm "$CONTAINER_NAME"
                ;;
            5)
                exit 0
                ;;
        esac
    else
        echo "Container exists but is not running"
        echo "Do you want to:"
        echo "  1) Start the existing container"
        echo "  2) Remove and create a new container"
        echo "  3) Exit"
        read -p "Enter choice [1-3]: " choice
        
        case $choice in
            1)
                echo -e "${BLUE}Starting existing container...${NC}"
                docker start "$CONTAINER_NAME"
                docker exec -it "$CONTAINER_NAME" /bin/bash
                exit 0
                ;;
            2)
                echo -e "${BLUE}Removing existing container...${NC}"
                docker rm "$CONTAINER_NAME"
                ;;
            3)
                exit 0
                ;;
        esac
    fi
fi

# Start new container
echo -e "${BLUE}Starting new container...${NC}"
echo -e "Image: ${GREEN}$IMAGE_NAME${NC}"
echo -e "Container name: ${GREEN}$CONTAINER_NAME${NC}"
echo -e "Mounted directory: ${GREEN}$SLIME_DIR${NC} -> ${GREEN}/workspace/slime${NC}"
echo ""

docker run -it \
    --name "$CONTAINER_NAME" \
    --hostname slime-dev \
    $GPU_FLAGS \
    --network host \
    --ipc host \
    --shm-size=16g \
    -v "$SLIME_DIR:/workspace/slime" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=all \
    -e PYTHONUNBUFFERED=1 \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e DISPLAY="${DISPLAY:-}" \
    -w /workspace/slime \
    "$IMAGE_NAME" \
    /bin/bash

echo -e "${GREEN}Container exited${NC}"



