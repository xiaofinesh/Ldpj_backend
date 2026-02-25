#!/usr/bin/env bash
# ============================================================================
# Ldpj_backend 离线部署脚本
# 适用于无法联网的 Linux 工控机 (x86_64, Python 3.11+)
#
# 使用方法:
#   1. 将整个 deploy/ 目录通过U盘拷贝到工控机
#   2. cd /path/to/deploy
#   3. sudo bash offline_install.sh
#
# 本脚本将:
#   - 检查 Python 3.11 环境
#   - 创建虚拟环境
#   - 从本地 wheel 包安装所有依赖 (无需联网)
#   - 部署项目到 /opt/ldpj_backend
#   - 创建 systemd 服务 (可选)
#   - 创建快捷启动脚本
# ============================================================================

set -euo pipefail

# ── 颜色输出 ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 配置变量 ────────────────────────────────────────────────────────────
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="/opt/ldpj_backend"
SERVICE_NAME="ldpj_backend"
VENV_DIR="${INSTALL_DIR}/.venv"
PACKAGES_DIR="${DEPLOY_DIR}/offline_packages"
PROJECT_DIR="${DEPLOY_DIR}/.."

# ── 0. 权限检查 ─────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    error "请使用 root 权限运行: sudo bash $0"
    exit 1
fi

echo ""
echo "============================================"
echo "  Ldpj_backend 离线部署工具 v2.1"
echo "============================================"
echo ""
info "部署源目录: ${DEPLOY_DIR}"
info "安装目标:   ${INSTALL_DIR}"
echo ""

# ── 1. 检查 Python 版本 ─────────────────────────────────────────────────
info "=== 步骤 1/7: 检查 Python 环境 ==="

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
    error "未找到 Python 3.11+!"
    echo ""
    echo "请先安装 Python 3.11，方法如下:"
    echo ""
    echo "  方法A (离线 deb 包):"
    echo "    将 python3.11 的 deb 包拷贝到工控机后执行:"
    echo "    sudo dpkg -i python3.11*.deb"
    echo ""
    echo "  方法B (源码编译):"
    echo "    将 Python-3.11.x.tar.xz 拷贝到工控机后执行:"
    echo "    tar xf Python-3.11.*.tar.xz"
    echo "    cd Python-3.11.*"
    echo "    ./configure --prefix=/usr/local --enable-optimizations"
    echo "    make -j\$(nproc) && sudo make altinstall"
    echo ""
    exit 1
fi

info "Python 版本: $($PYTHON --version)"

# 检查 venv 模块
if ! "$PYTHON" -c "import venv" 2>/dev/null; then
    error "Python venv 模块不可用!"
    echo "请安装: sudo apt install python3.11-venv (需要离线 deb 包)"
    exit 1
fi
info "venv 模块: 可用"

# ── 2. 检查离线包 ───────────────────────────────────────────────────────
echo ""
info "=== 步骤 2/7: 检查离线安装包 ==="

if [ ! -d "$PACKAGES_DIR" ]; then
    error "离线包目录不存在: $PACKAGES_DIR"
    exit 1
fi

PKG_COUNT=$(ls "$PACKAGES_DIR"/*.whl 2>/dev/null | wc -l)
if [ "$PKG_COUNT" -eq 0 ]; then
    error "离线包目录中没有 .whl 文件"
    exit 1
fi
info "找到 ${PKG_COUNT} 个离线 wheel 包"

# ── 3. 创建安装目录 ─────────────────────────────────────────────────────
echo ""
info "=== 步骤 3/7: 创建安装目录 ==="

if [ -d "$INSTALL_DIR" ]; then
    warn "安装目录已存在: $INSTALL_DIR"
    # 备份旧版本
    BACKUP="${INSTALL_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    info "备份旧版本到: $BACKUP"
    mv "$INSTALL_DIR" "$BACKUP"
fi

mkdir -p "$INSTALL_DIR"
info "创建目录: $INSTALL_DIR"

# ── 4. 复制项目文件 ─────────────────────────────────────────────────────
echo ""
info "=== 步骤 4/7: 复制项目文件 ==="

# 复制项目代码 (排除 deploy 目录、.git、__pycache__)
rsync -a \
    --exclude='deploy/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    "$PROJECT_DIR/" "$INSTALL_DIR/"

# 创建必要的子目录
mkdir -p "$INSTALL_DIR/models/artifacts/current"
mkdir -p "$INSTALL_DIR/models/artifacts/archive"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/data"

info "项目文件已复制到 $INSTALL_DIR"

# ── 5. 创建虚拟环境并安装依赖 ───────────────────────────────────────────
echo ""
info "=== 步骤 5/7: 创建虚拟环境并安装依赖 ==="

"$PYTHON" -m venv "$VENV_DIR"
info "虚拟环境已创建: $VENV_DIR"

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 升级 pip (使用离线包中的 pip 如果有，否则跳过)
if ls "$PACKAGES_DIR"/pip-*.whl 1>/dev/null 2>&1; then
    pip install --no-index --find-links="$PACKAGES_DIR" pip 2>/dev/null || true
fi

# 安装所有依赖 (完全离线)
info "正在安装 Python 依赖 (离线模式)..."
pip install --no-index --find-links="$PACKAGES_DIR" \
    numpy pandas scikit-learn joblib xgboost \
    python-snap7 PyYAML fastapi uvicorn httpx 2>&1 | \
    grep -E "^(Successfully|ERROR)" || true

# 验证安装
info "验证已安装的包:"
"$PYTHON" -c "
import sys
packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'joblib': 'joblib',
    'xgboost': 'xgboost',
    'snap7': 'python-snap7',
    'yaml': 'PyYAML',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'httpx': 'httpx',
}
ok = True
for mod, name in packages.items():
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'OK')
        print(f'  ✓ {name:20s} {ver}')
    except ImportError:
        print(f'  ✗ {name:20s} 未安装!')
        ok = False
if not ok:
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    error "部分依赖安装失败，请检查离线包是否完整"
    exit 1
fi

deactivate
info "所有依赖安装成功"

# ── 6. 创建启动脚本 ─────────────────────────────────────────────────────
echo ""
info "=== 步骤 6/7: 创建启动脚本 ==="

# 启动脚本 (生产模式)
cat > "$INSTALL_DIR/start.sh" <<'STARTEOF'
#!/usr/bin/env bash
# Ldpj_backend 启动脚本 (生产模式)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
cd "$SCRIPT_DIR"
exec python main.py --mode s7 "$@"
STARTEOF
chmod +x "$INSTALL_DIR/start.sh"

# 启动脚本 (开发/测试模式)
cat > "$INSTALL_DIR/start_mock.sh" <<'MOCKEOF'
#!/usr/bin/env bash
# Ldpj_backend 启动脚本 (Mock 测试模式)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
cd "$SCRIPT_DIR"
exec python main.py --mode mock "$@"
MOCKEOF
chmod +x "$INSTALL_DIR/start_mock.sh"

# 停止脚本
cat > "$INSTALL_DIR/stop.sh" <<'STOPEOF'
#!/usr/bin/env bash
# Ldpj_backend 停止脚本
PID=$(pgrep -f "python.*main.py" || true)
if [ -n "$PID" ]; then
    echo "正在停止 Ldpj_backend (PID: $PID)..."
    kill "$PID"
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "强制终止..."
        kill -9 "$PID"
    fi
    echo "已停止"
else
    echo "Ldpj_backend 未在运行"
fi
STOPEOF
chmod +x "$INSTALL_DIR/stop.sh"

info "启动脚本已创建:"
info "  生产模式: $INSTALL_DIR/start.sh"
info "  测试模式: $INSTALL_DIR/start_mock.sh"
info "  停止服务: $INSTALL_DIR/stop.sh"

# ── 7. 创建 systemd 服务 (可选) ─────────────────────────────────────────
echo ""
info "=== 步骤 7/7: 配置 systemd 服务 ==="

cat > /etc/systemd/system/${SERVICE_NAME}.service <<SVCEOF
[Unit]
Description=Ldpj_backend Edge AI Leak Detection System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
ExecStart=${VENV_DIR}/bin/python ${INSTALL_DIR}/main.py --mode s7
ExecStop=/bin/kill -SIGTERM \$MAINPID
Restart=on-failure
RestartSec=10
StandardOutput=append:${INSTALL_DIR}/logs/stdout.log
StandardError=append:${INSTALL_DIR}/logs/stderr.log

# 环境变量
Environment=PYTHONUNBUFFERED=1

# 资源限制
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
info "systemd 服务已创建: ${SERVICE_NAME}.service"
info "  启动服务: sudo systemctl start ${SERVICE_NAME}"
info "  停止服务: sudo systemctl stop ${SERVICE_NAME}"
info "  开机自启: sudo systemctl enable ${SERVICE_NAME}"
info "  查看状态: sudo systemctl status ${SERVICE_NAME}"
info "  查看日志: sudo journalctl -u ${SERVICE_NAME} -f"

# ── 完成 ────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
info "部署完成!"
echo "============================================"
echo ""
echo "安装位置: $INSTALL_DIR"
echo ""
echo "下一步操作:"
echo "  1. 编辑 PLC 配置:  nano $INSTALL_DIR/configs/plc.yaml"
echo "     - 修改 PLC IP 地址"
echo "     - 确认舱室数量 (cabin_count)"
echo ""
echo "  2. 部署 AI 模型 (如果有):"
echo "     bash $INSTALL_DIR/scripts/deploy_model.sh <模型目录>"
echo ""
echo "  3. 测试运行 (Mock 模式):"
echo "     $INSTALL_DIR/start_mock.sh"
echo ""
echo "  4. 生产运行:"
echo "     sudo systemctl start ${SERVICE_NAME}"
echo "     sudo systemctl enable ${SERVICE_NAME}  # 开机自启"
echo ""
echo "  5. 查看日志:"
echo "     tail -f $INSTALL_DIR/logs/stdout.log"
echo ""
