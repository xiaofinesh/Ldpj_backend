#!/usr/bin/env bash
# ============================================================================
# Ldpj_backend v2.5 离线部署脚本
# 适用于无法联网的 Linux 工控机 (x86_64 / aarch64, Python 3.11+)
#
# 使用方法:
#   1. 在联网开发机上准备离线包 (见 README.md 第2节)
#   2. 将整个项目目录通过U盘拷贝到工控机
#   3. cd Ldpj_backend/deploy && sudo bash offline_install.sh
# ============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="/opt/ldpj_backend"
SERVICE_NAME="ldpj_backend"
VENV_DIR="${INSTALL_DIR}/.venv"
PACKAGES_DIR="${DEPLOY_DIR}/offline_packages"
PROJECT_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"

# ── 0. 权限 ─────────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    error "请使用 root 权限: sudo bash $0"
    exit 1
fi

echo ""
echo "============================================"
echo "  Ldpj_backend v2.5 离线部署"
echo "============================================"
info "源目录: $PROJECT_DIR"
info "安装到: $INSTALL_DIR"
echo ""

# ── 1. Python ────────────────────────────────────────────────────────────
info "=== 步骤 1/7: 检查 Python ==="
PYTHON=""
for cmd in python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then PYTHON="$cmd"; break; fi
    fi
done
if [ -z "$PYTHON" ]; then
    error "未找到 Python 3.11+! 请先安装."
    exit 1
fi
info "Python: $($PYTHON --version)"
if ! "$PYTHON" -c "import venv" 2>/dev/null; then
    error "venv 模块不可用! 请安装: sudo apt install python3.11-venv"
    exit 1
fi

# ── 2. 离线包 ────────────────────────────────────────────────────────────
echo ""
info "=== 步骤 2/7: 检查离线包 ==="
if [ ! -d "$PACKAGES_DIR" ]; then
    error "离线包目录不存在: $PACKAGES_DIR"
    echo "请先在联网开发机上执行:"
    echo "  mkdir -p deploy/offline_packages"
    echo "  pip download -r requirements.txt -d deploy/offline_packages"
    exit 1
fi
PKG_COUNT=$(ls "$PACKAGES_DIR"/*.whl 2>/dev/null | wc -l)
info "找到 ${PKG_COUNT} 个 wheel 包"

# ── 3. 安装目录 ──────────────────────────────────────────────────────────
echo ""
info "=== 步骤 3/7: 创建安装目录 ==="
if [ -d "$INSTALL_DIR" ]; then
    BACKUP="${INSTALL_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    warn "备份旧版本: $BACKUP"
    mv "$INSTALL_DIR" "$BACKUP"
fi
mkdir -p "$INSTALL_DIR"

# ── 4. 复制项目 ──────────────────────────────────────────────────────────
echo ""
info "=== 步骤 4/7: 复制项目文件 ==="
rsync -a --exclude='deploy/' --exclude='.git/' --exclude='__pycache__/' \
    --exclude='*.pyc' --exclude='.venv/' "$PROJECT_DIR/" "$INSTALL_DIR/"
mkdir -p "$INSTALL_DIR/models/artifacts/current"
mkdir -p "$INSTALL_DIR/models/artifacts/archive"
mkdir -p "$INSTALL_DIR/logs"
info "已复制到 $INSTALL_DIR"

# ── 5. 虚拟环境 + 依赖 ──────────────────────────────────────────────────
echo ""
info "=== 步骤 5/7: 安装 Python 依赖 ==="
"$PYTHON" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --no-index --find-links="$PACKAGES_DIR" pip 2>/dev/null || true
pip install --no-index --find-links="$PACKAGES_DIR" \
    numpy pandas scikit-learn joblib xgboost \
    python-snap7 PyYAML fastapi uvicorn httpx 2>&1 | \
    grep -E "^(Successfully|ERROR)" || true
deactivate
info "依赖安装完成"

# ── 6. 启动脚本 ──────────────────────────────────────────────────────────
echo ""
info "=== 步骤 6/7: 创建启动脚本 ==="

cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
cd "$SCRIPT_DIR"
exec python main.py --mode s7 "$@"
EOF

cat > "$INSTALL_DIR/start_mock.sh" << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
cd "$SCRIPT_DIR"
exec python main.py --mode mock "$@"
EOF

cat > "$INSTALL_DIR/stop.sh" << 'EOF'
#!/usr/bin/env bash
PID=$(pgrep -f "python.*main.py" || true)
if [ -n "$PID" ]; then
    echo "停止 Ldpj_backend (PID: $PID)..."
    kill "$PID"; sleep 2
    kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
    echo "已停止"
else
    echo "未在运行"
fi
EOF

chmod +x "$INSTALL_DIR"/{start,start_mock,stop}.sh
chmod +x "$INSTALL_DIR/scripts/"*.sh 2>/dev/null || true

# ── 7. systemd ──────────────────────────────────────────────────────────
echo ""
info "=== 步骤 7/7: 配置 systemd ==="
cat > /etc/systemd/system/${SERVICE_NAME}.service << SVCEOF
[Unit]
Description=Ldpj_backend Edge AI Leak Detection
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
ExecStart=${VENV_DIR}/bin/python ${INSTALL_DIR}/main.py --mode s7
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SVCEOF
systemctl daemon-reload
info "systemd 服务: ${SERVICE_NAME}.service"

echo ""
echo "============================================"
info "部署完成!"
echo "============================================"
echo ""
echo "下一步:"
echo "  1. 配置PLC: nano $INSTALL_DIR/configs/plc.yaml"
echo "  2. Mock测试: $INSTALL_DIR/start_mock.sh"
echo "  3. 生产运行: sudo systemctl start $SERVICE_NAME"
echo "  4. 开机自启: sudo systemctl enable $SERVICE_NAME"
echo ""
