#!/bin/bash
set -e

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔨 Building TorchServe Multi-Device Image${NC}\n"

# Détection de l'architecture
ARCH=$(uname -m)
echo -e "${BLUE}📊 System Architecture: ${ARCH}${NC}"

# Sélection du compose file
if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    COMPOSE_OVERRIDE="docker-compose.arm64.yml"
    DEVICE="MPS (Apple Silicon GPU)"
    echo -e "${GREEN}✅ Detected Apple Silicon → Using ${COMPOSE_OVERRIDE}${NC}"
elif command -v nvidia-smi &> /dev/null; then
    COMPOSE_OVERRIDE="docker-compose.gpu.yml"
    DEVICE="CUDA GPU"
    echo -e "${GREEN}✅ Detected NVIDIA GPU → Using ${COMPOSE_OVERRIDE}${NC}"
else
    COMPOSE_OVERRIDE="docker-compose.cpu.yml"
    DEVICE="CPU"
    echo -e "${YELLOW}⚠️  No GPU detected → Using ${COMPOSE_OVERRIDE}${NC}"
fi

# Navigation vers le projet
cd "$(dirname "$0")/../.." || exit 1
PROJECT_DIR=$(pwd)
echo -e "${BLUE}📁 Project directory: ${PROJECT_DIR}${NC}\n"

# Vérification des fichiers requis
echo -e "${BLUE}📋 Checking required files...${NC}"
REQUIRED_FILES=(
    "backend/Dockerfile.torchserve"
    "backend/torchserve/start.sh"
    "backend/torchserve/handlers/sign_handler.py"
    "backend/torchserve/config/config.properties"
    "docker-compose.yml"
    "${COMPOSE_OVERRIDE}"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file ${RED}(MISSING)${NC}"
        exit 1
    fi
done

echo -e "\n${BLUE}🏗️  Building Docker image (this may take 5-10 minutes)...${NC}\n"

# Build avec BuildKit pour cache layers
DOCKER_BUILDKIT=1 docker-compose -f docker-compose.yml -f "${COMPOSE_OVERRIDE}" build torchserve

if [[ $? -eq 0 ]]; then
    echo -e "\n${GREEN}✅ Build successful!${NC}\n"
    echo -e "${BLUE}📊 Image Info:${NC}"
    docker images | grep signflow_torchserve || docker images | grep signflow | grep torchserve

    echo -e "\n${BLUE}🚀 Next Steps:${NC}"
    echo -e "  1. Start TorchServe:"
    echo -e "     ${YELLOW}docker-compose -f docker-compose.yml -f ${COMPOSE_OVERRIDE} up torchserve${NC}"
    echo -e ""
    echo -e "  2. Verify it's running:"
    echo -e "     ${YELLOW}curl http://localhost:8080/ping${NC}"
    echo -e ""
    echo -e "  3. View logs:"
    echo -e "     ${YELLOW}docker logs -f signflow_torchserve${NC}"
    echo -e ""
    echo -e "${BLUE}📚 Documentation:${NC}"
    echo -e "  - Quick Start: ${YELLOW}backend/GPU_QUICKSTART.md${NC}"
    echo -e "  - Full Guide:  ${YELLOW}backend/TORCHSERVE_MULTI_DEVICE.md${NC}"
    echo -e "  - Changelog:   ${YELLOW}backend/CHANGELOG_GPU.md${NC}"
else
    echo -e "\n${RED}❌ Build failed${NC}"
    exit 1
fi
