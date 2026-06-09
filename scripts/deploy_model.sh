#!/usr/bin/env bash
# deploy_model.sh – Deploy a trained v2.6 model bundle to the edge device.
#
# Usage:
#   bash scripts/deploy_model.sh <artifact_dir>
#
# Example:
#   bash scripts/deploy_model.sh models/artifacts/v2.6.2-cal20260605
#   # or straight from the calibration archive's PLC部署/ folder
#
# This script (v2.6 — dual-track M1 + M2 regression):
#   1. Validates the artifact dir contains the M1 coefficient table and the
#      M2 three-piece bundle (model + scaler + metadata).
#   2. Reminds the operator about the v2.6 PLC field-semantics change.
#   3. Archives the current model (if any).
#   4. Copies the new bundle into models/artifacts/current/.
#   5. Regenerates configs/models.yaml with matching m1/m2 blocks.
#   6. Prompts the user to restart the backend service.
#
# NOTE (v2.6): the old v2.5 binary classifier (xgb_model.json + xgb_scaler
# + metadata.json) is fully deprecated and NO LONGER deployed. The runtime
# loads M1 (per-cabin linear) + M2 (global XGBoost regression) only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CURRENT_DIR="$PROJECT_DIR/models/artifacts/current"
ARCHIVE_DIR="$PROJECT_DIR/models/artifacts/archive"
MODELS_YAML="$PROJECT_DIR/configs/models.yaml"

# v2.6 required bundle. m1_coefficients_25cabins.csv is a human-readable
# companion view and is copied if present but not required.
REQUIRED_FILES=(
    "m1_coefficients.json"
    "m2_xgb_model.json"
    "m2_xgb_scaler.joblib"
    "m2_metadata.json"
)

# ── Argument check ──────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/deploy_model.sh <artifact_dir>"
    exit 1
fi

ARTIFACT_DIR="$1"
if [ ! -d "$ARTIFACT_DIR" ]; then
    ARTIFACT_DIR="$PROJECT_DIR/$1"   # try relative to project root
fi
if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "ERROR: Artifact directory not found: $1"
    exit 1
fi

# ── 1. Validate artifacts ──────────────────────────────────────────────
echo "=== Step 1: Validating v2.6 artifacts ==="
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$ARTIFACT_DIR/$f" ]; then
        echo "ERROR: Missing required file: $f"
        echo "  (expected the M1 table + M2 three-piece bundle in $ARTIFACT_DIR)"
        exit 1
    fi
done
echo "All required files present."

# Extract version from m1_coefficients.json (authoritative for the bundle)
VERSION=$(python3 -c "import json; print(json.load(open('$ARTIFACT_DIR/m1_coefficients.json'))['version'])" 2>/dev/null || echo "unknown")
echo "Model version: $VERSION"

# Sanity-check that M1/M2 versions agree (warn only).
M2_VERSION=$(python3 -c "import json; print(json.load(open('$ARTIFACT_DIR/m2_metadata.json')).get('version','?'))" 2>/dev/null || echo "?")
if [ "$VERSION" != "$M2_VERSION" ]; then
    echo "WARNING: M1 version ($VERSION) != M2 version ($M2_VERSION); deploying anyway."
fi

# ── 1c. v2.6.3 工况(operating_point)兼容性校验 ─────────────────────────
# 工件标定工况必须与目标 runtime.yaml 活动工况兼容, 否则部署后启动会被
# 工况门拒绝。几何/身份/采样数变化 → 拒绝部署; 仅间隔(采样数保持)或真空
# 不一致 → 放行(运行期自动重缩放/告警)。
echo ""
echo "=== Step 1c: 工况指纹兼容性校验 ==="
set +e
ARTIFACT_DIR="$ARTIFACT_DIR" PROJECT_DIR="$PROJECT_DIR" python3 <<'PYEOF'
import os, sys, json
sys.path.insert(0, os.environ["PROJECT_DIR"])
art_dir = os.environ["ARTIFACT_DIR"]
try:
    from core.operating_point import OperatingPoint
    from configs.loaders import load_active_cycle_profile, validate_operating_point
except Exception as e:
    print(f"  跳过(无法导入, 非阻断): {e}"); sys.exit(0)
m1 = json.load(open(os.path.join(art_dir, "m1_coefficients.json"), encoding="utf-8"))
op = m1.get("operating_point")
if not op:
    print("  ERROR: m1_coefficients.json 缺少 operating_point 工况指纹 (需 v2.6.3+ 训练/回填)")
    sys.exit(3)
m1_op = OperatingPoint.from_fingerprint(op)
m2_op = None
m2p = os.path.join(art_dir, "m2_metadata.json")
if os.path.exists(m2p):
    m2 = json.load(open(m2p, encoding="utf-8"))
    if m2.get("operating_point"):
        m2_op = OperatingPoint.from_fingerprint(m2["operating_point"])
active = load_active_cycle_profile().operating_point()
r = validate_operating_point(active, m1_op, m2_op)
for w in r["warnings"]:
    print("  WARN:", w)
if r["errors"]:
    for e in r["errors"]:
        print("  ERROR:", e)
    print("  工件标定工况与目标 runtime.yaml 活动工况不兼容; 部署被拒绝。")
    sys.exit(3)
print(f"  OK: 工件工况 [{m1_op.profile_id}] 与目标 runtime 兼容"
      f"{' (运行期将重缩放/告警)' if (r['m1_rescale_to'] or r['warnings']) else ''}。")
sys.exit(0)
PYEOF
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
    echo "部署中止 (工况校验失败)。如确为新工况, 请先更新 runtime.yaml 的 active_profile。"
    exit 1
fi

# ── 1b. v2.6 PLC 字段语义变更确认 ──────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ⚠ v2.6 PLC 字段语义变更 + 分段前置条件"
echo "═══════════════════════════════════════════════════════════════"
echo "  cabinHealthStatus (REAL @ DB9 +14) 在 v2.6 起承载 Q_est"
echo "  单位: Pa·m³/s   典型范围: 1e-7 ~ 1e-2   (v2.5 为 [0,1] 概率)"
echo "  字节格式不变, 但 HMI 显示逻辑必须同步切换。"
echo ""
echo "  另: 部署前 runtime.yaml 保压窗必须为 [93°,283°) —— 否则系数表失配。"
echo ""
if [ -n "${LDPJ_SKIP_HMI_CONFIRM:-}" ]; then
    echo "  [LDPJ_SKIP_HMI_CONFIRM 已设置, 跳过确认]"
else
    read -r -p "  HMI 已确认支持 Q_est 显示, 且分段已为 [93,283)? (y/N): " hmi_ack
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
    OLD_VERSION=$(python3 -c "import json; print(json.load(open('$CURRENT_DIR/m1_coefficients.json'))['version'])" 2>/dev/null || echo "old")
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
for f in "${REQUIRED_FILES[@]}"; do
    cp "$ARTIFACT_DIR/$f" "$CURRENT_DIR/"
done
# Optional companion files
[ -f "$ARTIFACT_DIR/m1_coefficients_25cabins.csv" ] && cp "$ARTIFACT_DIR/m1_coefficients_25cabins.csv" "$CURRENT_DIR/"
[ -f "$ARTIFACT_DIR/evaluation_report.txt" ] && cp "$ARTIFACT_DIR/evaluation_report.txt" "$CURRENT_DIR/"
echo "Deployed to: $CURRENT_DIR"

# ── 4. Update models.yaml ──────────────────────────────────────────────
echo ""
echo "=== Step 4: Updating configuration ==="
cat > "$MODELS_YAML" <<EOF
# Model management configuration (auto-updated by deploy_model.sh)
#
# v2.6: dual-track regression (M1 per-cabin linear + M2 global XGBoost).
# The v2.5 binary classifier is fully deprecated and no longer referenced.

# M1 — per-cabin linear regression (primary Q estimator)
m1:
  version: "$VERSION"
  coefficients_path: "models/artifacts/current/m1_coefficients.json"

# M2 — global XGBoost regression on 36-dim features (cross-check, F010)
m2:
  version: "$VERSION"
  model_path: "models/artifacts/current/m2_xgb_model.json"
  scaler_path: "models/artifacts/current/m2_xgb_scaler.joblib"
  metadata_path: "models/artifacts/current/m2_metadata.json"

archive_dir: "models/artifacts/archive"
EOF
echo "Updated: $MODELS_YAML"

# ── 5. Done ─────────────────────────────────────────────────────────────
echo ""
echo "=== Deployment complete ==="
echo "Model bundle '$VERSION' is now active (M1 + M2)."
echo ""
echo "Please restart the backend service to load the new model:"
echo "  sudo systemctl restart ldpj_backend"
echo "  # or: python main.py --mode s7"
