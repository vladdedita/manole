#!/usr/bin/env bash
set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${BLUE}[*]${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}[+]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[!]${NC} %s\n" "$*"; }
err()   { printf "${RED}[x]${NC} %s\n" "$*" >&2; }
step()  { printf "\n${BOLD}── %s ──${NC}\n" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── OS / arch detection ────────────────────────────────────────────
detect_os() {
  case "$(uname -s)" in
    Darwin) OS=macos ;;
    Linux)  OS=linux ;;
    *)      err "Unsupported OS: $(uname -s)"; exit 1 ;;
  esac
  case "$(uname -m)" in
    arm64|aarch64) ARCH=arm64 ;;
    x86_64)        ARCH=x86_64 ;;
    *)             err "Unsupported architecture: $(uname -m)"; exit 1 ;;
  esac
}

# ── Version helpers ────────────────────────────────────────────────
# Returns 0 if $1 >= $2 (dot-separated version comparison)
version_gte() {
  printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1 | grep -qx "$2"
}

# ── Dependency checks (idempotent) ─────────────────────────────────

ensure_homebrew() {
  [[ "$OS" != macos ]] && return
  if command -v brew &>/dev/null; then
    ok "Homebrew found"
    return
  fi
  warn "Homebrew not found. It's needed to install dependencies on macOS."
  read -rp "Install Homebrew now? [Y/n] " answer
  if [[ "${answer:-Y}" =~ ^[Yy]$ ]]; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Ensure brew is on PATH for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    ok "Homebrew installed"
  else
    err "Homebrew is required on macOS. Aborting."
    exit 1
  fi
}

ensure_python() {
  local required="3.13"
  if command -v python3.13 &>/dev/null; then
    ok "Python 3.13 found: $(python3.13 --version)"
    return
  fi
  if command -v python3 &>/dev/null; then
    local ver
    ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if version_gte "$ver" "$required"; then
      ok "Python $ver found (>= $required)"
      return
    fi
  fi
  info "Installing Python $required..."
  if [[ "$OS" == macos ]]; then
    brew install python@3.13
  else
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.13 python3.13-venv
  fi
  ok "Python 3.13 installed"
}

ensure_node() {
  local required="20"
  if command -v node &>/dev/null; then
    local ver
    ver="$(node -v | sed 's/^v//' | cut -d. -f1)"
    if [[ "$ver" -ge "$required" ]]; then
      ok "Node.js v$(node -v | sed 's/^v//') found (>= $required)"
      return
    fi
    warn "Node.js v$(node -v | sed 's/^v//') is too old (need >= $required)"
  fi
  info "Installing Node.js..."
  if [[ "$OS" == macos ]]; then
    brew install node
  else
    # NodeSource setup for Node 22 LTS
    if ! command -v curl &>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y -qq curl
    fi
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y -qq nodejs
  fi
  ok "Node.js $(node -v | sed 's/^v//') installed"
}

ensure_uv() {
  if command -v uv &>/dev/null; then
    ok "uv found: $(uv --version)"
    return
  fi
  info "Installing uv..."
  if [[ "$OS" == macos ]]; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  ok "uv installed: $(uv --version)"
}

# ── Project setup (idempotent) ──────────────────────────────────────

setup_python_env() {
  step "Python environment"
  if [[ -d .venv ]] && [[ -f uv.lock ]]; then
    info "Syncing Python dependencies..."
  else
    info "Creating Python environment and installing dependencies..."
  fi
  uv sync
  ok "Python environment ready"
}

setup_node_env() {
  step "Node environment"
  if [[ -d ui/node_modules ]]; then
    info "Syncing Node dependencies..."
  else
    info "Installing Node dependencies (this also builds the Electron app)..."
  fi
  (cd ui && npm install)
  ok "Node environment ready"
}

# ── Model download (idempotent) ────────────────────────────────────

ensure_models() {
  step "AI models"
  local models_dir="$SCRIPT_DIR/models"
  local manifest="$SCRIPT_DIR/models-manifest.json"
  local all_present=true

  mkdir -p "$models_dir"

  # Parse manifest and check/download each required model
  local count
  count="$(.venv/bin/python3 -c "
import json
m = json.load(open('$manifest'))
print(len([x for x in m['models'] if x.get('required', False)]))
")"

  for i in $(seq 0 $((count - 1))); do
    local model_info
    model_info="$(.venv/bin/python3 -c "
import json
m = json.load(open('$manifest'))
models = [x for x in m['models'] if x.get('required', False)]
e = models[$i]
print(f\"{e['id']}|{e['filename']}|{e['repo_id']}\")
")"

    local model_id filename repo_id
    model_id="$(echo "$model_info" | cut -d'|' -f1)"
    filename="$(echo "$model_info" | cut -d'|' -f2)"
    repo_id="$(echo "$model_info" | cut -d'|' -f3)"

    if [[ -f "$models_dir/$filename" ]]; then
      ok "$model_id: already downloaded"
    else
      all_present=false
      info "$model_id: downloading $filename from $repo_id..."
      .venv/bin/python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$repo_id', filename='$filename', local_dir='$models_dir')
"
      ok "$model_id: download complete"
    fi
  done

  if [[ "$all_present" == true ]]; then
    ok "All models present"
  else
    ok "Model downloads finished"
  fi
}

# ── Launch ──────────────────────────────────────────────────────────

launch() {
  step "Launching Manole"
  info "Starting Electron + Python backend..."
  info "Press Cmd+Q (macOS) or close the window to exit."
  echo
  cd ui && exec npm run dev
}

# ── Main ────────────────────────────────────────────────────────────

main() {
  echo
  printf "${BOLD}Manole — local AI file assistant${NC}\n"
  echo

  detect_os
  info "Detected: $OS / $ARCH"

  step "Checking prerequisites"
  ensure_homebrew
  ensure_python
  ensure_node
  ensure_uv

  setup_python_env
  ensure_models
  setup_node_env
  launch
}

main "$@"
