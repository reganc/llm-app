#!/usr/bin/env bash
# =============================================================
#  diagnose.sh — Pre-flight GPU + Docker checks
#  Run this BEFORE docker compose up to catch issues early
# =============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✔${NC}  $*"; }
fail() { echo -e "  ${RED}✘${NC}  $*"; FAILED=1; }
info() { echo -e "  ${BLUE}→${NC}  $*"; }
FAILED=0

echo ""
echo -e "${BLUE}══════════════════════════════════════════${NC}"
echo -e "${BLUE}   LLM Stack Pre-flight Diagnostics       ${NC}"
echo -e "${BLUE}══════════════════════════════════════════${NC}"
echo ""

# ── 1. NVIDIA Driver ──────────────────────────────────────────
echo -e "${YELLOW}[1/5] NVIDIA Driver${NC}"
if command -v nvidia-smi &>/dev/null; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
  ok "Driver: $DRIVER"
  ok "GPU: $GPU"
  ok "VRAM: $VRAM"
else
  fail "nvidia-smi not found — install drivers: sudo apt install nvidia-driver-535"
fi

# ── 2. Docker ─────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/5] Docker${NC}"
if command -v docker &>/dev/null; then
  ok "Docker: $(docker --version)"
  if docker compose version &>/dev/null; then
    ok "Compose: $(docker compose version)"
  else
    fail "Docker Compose plugin missing: sudo apt install docker-compose-plugin"
  fi
  if groups "$USER" | grep -q docker || [ "$EUID" -eq 0 ]; then
    ok "User '$USER' is in docker group"
  else
    fail "User '$USER' not in docker group — run: sudo usermod -aG docker $USER && newgrp docker"
  fi
else
  fail "Docker not installed — run: curl -fsSL https://get.docker.com | sh"
fi

# ── 3. NVIDIA Container Toolkit ──────────────────────────────
echo ""
echo -e "${YELLOW}[3/5] NVIDIA Container Toolkit${NC}"
if dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit \
   || command -v nvidia-ctk &>/dev/null; then
  ok "nvidia-container-toolkit installed"
else
  info "nvidia-container-toolkit not found via dpkg (may be installed another way — GPU passthrough test below is the real proof)"
fi
# Check Docker runtime config regardless of package manager result
if docker info 2>/dev/null | grep -q "nvidia"; then
  ok "NVIDIA runtime registered with Docker"
else
  fail "NVIDIA runtime not in Docker — run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

# ── 4. GPU Passthrough Test ───────────────────────────────────
echo ""
echo -e "${YELLOW}[4/5] GPU Passthrough into Docker${NC}"
info "Running nvidia-smi inside a container (takes ~10s)..."
if docker run --rm --gpus all \
     nvidia/cuda:12.3.0-base-ubuntu22.04 \
     nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
  ok "GPU passthrough confirmed!"
else
  fail "GPU passthrough FAILED"
  echo ""
  echo -e "  ${YELLOW}Try these fixes in order:${NC}"
  echo "    1. sudo nvidia-ctk runtime configure --runtime=docker"
  echo "    2. sudo systemctl restart docker"
  echo "    3. Log out and back in (group membership)"
  echo "    4. Reboot: sudo reboot"
fi

# ── 5. Port Availability ──────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/5] Port Availability${NC}"
for port in 11434 8000 3000; do
  if ss -tlnp 2>/dev/null | grep -q ":$port " || netstat -tlnp 2>/dev/null | grep -q ":$port "; then
    fail "Port $port is already in use — change it in docker-compose.yml"
  else
    ok "Port $port is free"
  fi
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo -e "${BLUE}══════════════════════════════════════════${NC}"
if [ "$FAILED" -eq 0 ]; then
  echo -e "  ${GREEN}✅  All checks passed! Ready to launch.${NC}"
  echo ""
  echo -e "  Run: ${BLUE}docker compose up -d${NC}"
else
  echo -e "  ${RED}❌  Some checks failed. Fix the issues above, then re-run.${NC}"
  echo ""
  echo -e "  For full setup: ${BLUE}./scripts/setup.sh${NC}"
fi
echo -e "${BLUE}══════════════════════════════════════════${NC}"
echo ""
exit "$FAILED"
