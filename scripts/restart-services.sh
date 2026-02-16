#!/bin/bash
set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Restarting SignFlow Services${NC}\n"

# Navigation vers le projet
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
echo -e "${BLUE}📁 Project: ${PROJECT_DIR}${NC}\n"

# Détection architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.arm64.yml"
    echo -e "${GREEN}✅ Apple Silicon detected → Using MPS configuration${NC}"
else
    COMPOSE_FILES="-f docker-compose.yml"
    echo -e "${YELLOW}⚠️  x86_64 detected → Using default configuration${NC}"
fi

# Arrêt des services
echo -e "\n${BLUE}🛑 Stopping services...${NC}"
docker compose $COMPOSE_FILES down

# Rebuild TorchServe si nécessaire
if [[ "$1" == "--rebuild" ]]; then
    echo -e "\n${BLUE}🏗️  Rebuilding TorchServe...${NC}"
    docker compose $COMPOSE_FILES build torchserve
fi

# Démarrage
echo -e "\n${BLUE}🚀 Starting services...${NC}"
docker compose $COMPOSE_FILES up -d

# Attente que les services démarrent
echo -e "\n${BLUE}⏳ Waiting for services to start...${NC}"
sleep 5

# Vérification des services
echo -e "\n${BLUE}📊 Service Status:${NC}"
docker compose $COMPOSE_FILES ps

# Tests de santé
echo -e "\n${BLUE}🏥 Health Checks:${NC}"

# Backend
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Backend (http://localhost:8000)"
else
    echo -e "  ${RED}✗${NC} Backend (http://localhost:8000)"
fi

# Frontend
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Frontend (http://localhost:3000)"
else
    echo -e "  ${YELLOW}⏳${NC} Frontend (http://localhost:3000) - Still starting..."
fi

# TorchServe
if curl -s http://localhost:8080/ping > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} TorchServe (http://localhost:8080)"
else
    echo -e "  ${YELLOW}⏳${NC} TorchServe (http://localhost:8080) - Still starting..."
fi

# MLflow
if curl -s http://localhost:5001 > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} MLflow (http://localhost:5001)"
else
    echo -e "  ${YELLOW}⏳${NC} MLflow (http://localhost:5001) - Still starting..."
fi

echo -e "\n${GREEN}✅ Services restarted!${NC}"
echo -e "\n${BLUE}📚 Quick Links:${NC}"
echo -e "  Frontend:  ${YELLOW}http://localhost:3000${NC}"
echo -e "  Backend:   ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "  TorchServe: ${YELLOW}http://localhost:8080/ping${NC}"
echo -e "  MLflow:    ${YELLOW}http://localhost:5001${NC}"

echo -e "\n${BLUE}📝 View Logs:${NC}"
echo -e "  All:       ${YELLOW}docker compose $COMPOSE_FILES logs -f${NC}"
echo -e "  Frontend:  ${YELLOW}docker logs -f signflow-frontend-1${NC}"
echo -e "  Backend:   ${YELLOW}docker logs -f signflow-backend-1${NC}"
echo -e "  TorchServe: ${YELLOW}docker logs -f signflow_torchserve${NC}"

echo -e "\n${BLUE}🔧 Options:${NC}"
echo -e "  Rebuild:   ${YELLOW}$0 --rebuild${NC}"
