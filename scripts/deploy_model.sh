#!/usr/bin/env bash
# deploy_model.sh – Deploy a trained model to the edge device.
#
# Usage:
#   bash scripts/deploy_model.sh <artifact_dir>
#
# Example:
#   bash scripts/deploy_model.sh models/artifacts/v1.0_20260225
#
# This script:
#   1. Validates the artifact directory contains all required files.
#   2. Archives the current model (if any).
#   3. Copies the new model into models/artifacts/current/.
#   4. Updates configs/models.yaml with the new version.
#   5. Prompts the user to restart the backend service.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CURRENT_DIR="$PROJECT_DIR/models/artifacts/current"
ARCHIVE_DIR="$PROJECT_DIR/models/artifacts/archive"
MODELS_YAML="$PROJECT_DIR/configs/models.yaml"

REQUIRED_FILES=("xgb_model.json" "xgb_scaler.joblib" "metadata.json" "evaluation_report.txt")

# ── Argument check ──────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/deploy_model.sh <artifact_dir>"
    exit 1
fi

ARTIFACT_DIR="$1"
if [ ! -d "$ARTIFACT_DIR" ]; then
    # Try relative to project root
    ARTIFACT_DIR="$PROJECT_DIR/$1"
fi

if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "ERROR: Artifact directory not found: $1"
    exit 1
fi

# ── 1. Validate artifacts ──────────────────────────────────────────────
echo "=== Step 1: Validating artifacts ==="
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$ARTIFACT_DIR/$f" ]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
done
echo "All required files present."

# Extract version from metadata.json
VERSION=$(python3 -c "import json; print(json.load(open('$ARTIFACT_DIR/metadata.json'))['version'])" 2>/dev/null || echo "unknown")
echo "Model version: $VERSION"

# ── 1b. v2.6 PLC 字段语义变更确认 ──────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ⚠ v2.6 PLC 字段语义变更"
echo "═══════════════════════════════════════════════════════════════"
echo "  cabinHealthStatus (REAL @ DB9 +14) 在 v2.6 起承载 Q_est"
echo "  单位: Pa·m³/s   典型范围: 1e-7 ~ 1e-2"
echo "  v2.5 时该字段为 [0,1] 正常概率"
echo ""
echo "  字节格式不变, 但 HMI 显示逻辑必须同步切换。"
echo ""
if [ -n "${LDPJ_SKIP_HMI_CONFIRM:-}" ]; then
    echo "  [LDPJ_SKIP_HMI_CONFIRM 已设置, 跳过确认]"
else
    read -r -p "  HMI 已确认支持 Q_est 显示? (y/N): " hmi_ack
    case "${hmi_ack:-N}" in
        y|Y|yes|YES|Yes) ;;
        *)
            echo ""
            echo "  请先协调自控/HMI 团队后再继续部署。退出。"
            echo "  自动化场景可设置 LDPJ_SKIP_HMI_CONFIRM=1 跳过此提示。"
            exit 1
            ;;
    esac
fi

# ── 2. Archive current model ───────────────────────────────────────────
echo ""
echo "=== Step 2: Archiving current model ==="
mkdir -p "$ARCHIVE_DIR"
if [ -d "$CURRENT_DIR" ] && [ "$(ls -A "$CURRENT_DIR" 2>/dev/null)" ]; then
    OLD_VERSION=$(python3 -c "import json; print(json.load(open('$CURRENT_DIR/metadata.json'))['version'])" 2>/dev/null || echo "old")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVE_NAME="${OLD_VERSION}_${TIMESTAMP}"
    mv "$CURRENT_DIR" "$ARCHIVE_DIR/$ARCHIVE_NAME"
    echo "Archived: $ARCHIVE_DIR/$ARCHIVE_NAME"
else
    echo "No existing model to archive."
fi

# ── 3. Deploy new model ────────────────────────────────────────────────
echo ""
echo "=== Step 3: Deploying new model ==="
mkdir -p "$CURRENT_DIR"
cp "$ARTIFACT_DIR"/* "$CURRENT_DIR/"
echo "Deployed to: $CURRENT_DIR"

# ── 4. Update models.yaml ──────────────────────────────────────────────
echo ""
echo "=== Step 4: Updating configuration ==="
cat > "$MODELS_YAML" <<EOF
# Model management configuration (auto-updated by deploy_model.sh)

current:
  model_path: "models/artifacts/current/xgb_model.json"
  scaler_path: "models/artifacts/current/xgb_scaler.joblib"
  version: "$VERSION"
  loaded_at: null

archive_dir: "models/artifacts/archive"
EOF
echo "Updated: $MODELS_YAML"

# ── 5. Done ─────────────────────────────────────────────────────────────
echo ""
echo "=== Deployment complete ==="
echo "Model version '$VERSION' is now active."
echo ""
echo "Please restart the backend service to load the new model:"
echo "  sudo systemctl restart ldpj_backend"
echo "  # or: python main.py --mode s7"
