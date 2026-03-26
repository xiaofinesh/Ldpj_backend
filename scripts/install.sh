#!/usr/bin/env bash
# install.sh – 在线安装 Ldpj_backend (Debian/Ubuntu)
# 前提: Python 3.11+ 已安装
# 用法: bash scripts/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Ldpj_backend v2.5 安装 ==="
echo "项目目录: $PROJECT_DIR"

# ── 1. Python 版本检查 ─────────────────────────────────────────────────
echo ""
echo "--- 检查 Python ---"
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
    echo "错误: 需要 Python 3.11+ 但未找到."
    echo "安装: sudo apt install python3.11 python3.11-venv"
    exit 1
fi
echo "使用: $PYTHON ($($PYTHON --version))"

# ── 2. 虚拟环境 ────────────────────────────────────────────────────────
echo ""
echo "--- 创建虚拟环境 ---"
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
    echo "已创建: $VENV_DIR"
else
    echo "已存在: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 3. 安装依赖 ────────────────────────────────────────────────────────
echo ""
echo "--- 安装 Python 依赖 ---"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

# ── 4. snap7 系统库 ────────────────────────────────────────────────────
echo ""
echo "--- 检查 snap7 ---"
if ! ldconfig -p 2>/dev/null | grep -q libsnap7; then
    echo "安装 libsnap7..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq libsnap7-dev 2>/dev/null || {
        echo "警告: libsnap7 无法通过 apt 安装."
        echo "真实PLC模式需要手动安装 snap7: https://snap7.sourceforge.net/"
    }
else
    echo "libsnap7 已安装."
fi

# ── 5. 目录结构 ────────────────────────────────────────────────────────
echo ""
echo "--- 创建目录 ---"
mkdir -p "$PROJECT_DIR/models/artifacts/current"
mkdir -p "$PROJECT_DIR/models/artifacts/archive"
mkdir -p "$PROJECT_DIR/logs"
chmod +x "$PROJECT_DIR/scripts/"*.sh 2>/dev/null || true

# ── 6. 验证 ────────────────────────────────────────────────────────────
echo ""
echo "--- 验证安装 ---"
"$PYTHON" -c "
import yaml, numpy, pandas, sklearn, fastapi, uvicorn
print('  PyYAML:       ' + yaml.__version__)
print('  NumPy:        ' + numpy.__version__)
print('  pandas:       ' + pandas.__version__)
print('  scikit-learn: ' + sklearn.__version__)
print('  FastAPI:      ' + fastapi.__version__)
try:
    import xgboost; print('  XGBoost:      ' + xgboost.__version__)
except ImportError: print('  XGBoost:      未安装 (pip install xgboost)')
try:
    import snap7; print('  python-snap7: OK')
except ImportError: print('  python-snap7: 未安装 (仅Mock模式可用)')
"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "快速开始:"
echo "  source .venv/bin/activate"
echo "  python main.py --mode mock    # Mock测试模式"
echo "  python main.py --mode s7      # 生产模式 (需连接PLC)"
echo ""
echo "首次使用流程:"
echo "  1. 启动系统: python main.py --mode mock"
echo "  2. 输入 s 开始采集"
echo "  3. 等待数据积累后输入 x 导出CSV"
echo "  4. 训练: python -m train.train_model --data export.csv --output models/artifacts/v1.0"
echo "  5. 部署: bash scripts/deploy_model.sh models/artifacts/v1.0"
