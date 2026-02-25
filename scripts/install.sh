#!/usr/bin/env bash
# install.sh – Set up Ldpj_backend on a fresh Linux system (Debian/Ubuntu).
#
# Prerequisites: Python 3.11+ must be installed.
#
# Usage:
#   bash scripts/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Ldpj_backend Installation ==="
echo "Project directory: $PROJECT_DIR"

# ── 1. Check Python version ────────────────────────────────────────────
echo ""
echo "--- Checking Python ---"
PYTHON=""
for cmd in python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.11+ is required but not found."
    echo "Install with: sudo apt install python3.11 python3.11-venv"
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version))"

# ── 2. Create virtual environment ──────────────────────────────────────
echo ""
echo "--- Creating virtual environment ---"
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
    echo "Created: $VENV_DIR"
else
    echo "Exists: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── 3. Install dependencies ────────────────────────────────────────────
echo ""
echo "--- Installing dependencies ---"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

# ── 4. Install snap7 system library ────────────────────────────────────
echo ""
echo "--- Checking snap7 library ---"
if ! ldconfig -p 2>/dev/null | grep -q libsnap7; then
    echo "Installing libsnap7..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq libsnap7-dev 2>/dev/null || {
        echo "WARNING: libsnap7 not available via apt."
        echo "For real PLC mode, install snap7 manually:"
        echo "  https://snap7.sourceforge.net/"
    }
else
    echo "libsnap7 already installed."
fi

# ── 5. Create data directories ─────────────────────────────────────────
echo ""
echo "--- Creating directories ---"
mkdir -p "$PROJECT_DIR/models/artifacts/current"
mkdir -p "$PROJECT_DIR/models/artifacts/archive"
mkdir -p "$PROJECT_DIR/logs"

# ── 6. Make scripts executable ──────────────────────────────────────────
chmod +x "$PROJECT_DIR/scripts/"*.sh 2>/dev/null || true

# ── 7. Verify installation ─────────────────────────────────────────────
echo ""
echo "--- Verifying installation ---"
"$PYTHON" -c "
import yaml, numpy, pandas, sklearn, fastapi, uvicorn
print('  PyYAML:       ' + yaml.__version__)
print('  NumPy:        ' + numpy.__version__)
print('  pandas:       ' + pandas.__version__)
print('  scikit-learn: ' + sklearn.__version__)
print('  FastAPI:      ' + fastapi.__version__)
try:
    import xgboost
    print('  XGBoost:      ' + xgboost.__version__)
except ImportError:
    print('  XGBoost:      NOT INSTALLED (install with: pip install xgboost)')
try:
    import snap7
    print('  python-snap7: OK')
except ImportError:
    print('  python-snap7: NOT INSTALLED (mock mode only)')
"

echo ""
echo "=== Installation complete ==="
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  python main.py --mode mock    # Development mode"
echo "  python main.py --mode s7      # Production mode (requires PLC)"
