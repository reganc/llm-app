#!/usr/bin/env bash
# =============================================================
#  setup.sh — One-shot setup for Local LLM Stack on Ubuntu/Debian
#  RTX 3060 + Docker + Ollama + Open WebUI + Custom API
# =============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

echo -e "\n${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Local LLM Stack — RTX 3060 Setup      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}\n"

# ── 1. Check Ubuntu/Debian ────────────────────────────────────
if ! command -v apt-get &>/dev/null; then
  error "This script requires Ubuntu or Debian (apt-get not found)"
fi

# ── 2. Check NVIDIA GPU ───────────────────────────────────────
info "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
  error "nvidia-smi not found. Install NVIDIA drivers first:\n  sudo apt install nvidia-driver-535"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
success "GPU detected: ${GPU_NAME} (${VRAM})"

# ── 3. Install Docker ─────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  info "Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  warn "Added $USER to docker group. You may need to log out and back in."
else
  success "Docker already installed: $(docker --version)"
fi

# ── 4. Install Docker Compose ─────────────────────────────────
if ! docker compose version &>/dev/null; then
  info "Installing Docker Compose plugin..."
  sudo apt-get install -y docker-compose-plugin
else
  success "Docker Compose: $(docker compose version)"
fi

# ── 5. Install NVIDIA Container Toolkit ──────────────────────
info "Setting up NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update -qq
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  success "NVIDIA Container Toolkit installed"
else
  success "NVIDIA Container Toolkit already installed"
fi

# ── 6. Verify GPU passthrough ─────────────────────────────────
info "Verifying GPU passthrough to Docker..."
if docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
  success "GPU passthrough working!"
else
  error "GPU passthrough failed. Check: sudo nvidia-ctk runtime configure --runtime=docker"
fi

# ── 7. Pull images ────────────────────────────────────────────
info "Pulling Docker images (this may take a few minutes)..."
docker compose pull ollama open-webui

# ── 8. Launch stack ───────────────────────────────────────────
info "Starting services..."
docker compose up -d --build

# ── 9. Wait for Ollama ────────────────────────────────────────
info "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    success "Ollama is ready!"
    break
  fi
  echo -n "."
  sleep 2
done

# ── 10. Pull model ────────────────────────────────────────────
info "Pulling Mistral 7B Q4_K_M (~4.1 GB download)..."
docker exec ollama ollama pull mistral:7b-instruct-q4_K_M
success "Model downloaded and ready!"

# ── 11. Done ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅  Stack is running!                          ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  🌐  Web UI:   http://localhost:3000             ║${NC}"
echo -e "${GREEN}║  🔌  API:      http://localhost:8000             ║${NC}"
echo -e "${GREEN}║  📖  API Docs: http://localhost:8000/docs        ║${NC}"
echo -e "${GREEN}║  🔧  Ollama:   http://localhost:11434            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  API Key: ${YELLOW}change-me-in-production${NC} (set in docker-compose.yml)"
echo ""
echo -e "  Useful commands:"
echo -e "    ${BLUE}docker compose logs -f${NC}          # Follow all logs"
echo -e "    ${BLUE}docker compose down${NC}             # Stop everything"
echo -e "    ${BLUE}docker exec ollama ollama list${NC}  # List models"
echo -e "    ${BLUE}docker compose restart api${NC}      # Restart API only"
echo ""
