# Ldpj_backend

**基于XGBoost的真空衰减法漏液检测边缘计算后端系统 v2.1**

## 概述

本系统是一个部署在Linux工控机上的边缘AI后端，通过S7协议直连西门子PLC（CPU 1511-1 PN），高频采集6个测试舱室的真空压力曲线数据，利用XGBoost模型进行实时漏液检测推理，并将结果写回PLC供HMI显示。

## 系统架构

```
PLC (S7-1200/1500)
    │
    ├── DB9: Cabin[0..5] × {RT_AI, RT_Pressure, RT_Position, RT_Angle}
    │
    ▼
┌─────────────────────────────────────────────┐
│  Ldpj_backend (Linux / Python 3.11)         │
│                                             │
│  PollingEngine ──► CycleFSM ──► Features    │
│       │                            │        │
│       │                       XGBoost ──► ResultSender ──► PLC
│       │                            │        │
│       │                       DBLogger      │
│       │                                     │
│  HealthChecker ──► FaultReporter ──► AlarmPusher ──► 外部工控机
│                                             │
│  APIServer (FastAPI) ◄── 外部工控机数据查询  │
└─────────────────────────────────────────────┘
```

## 环境要求

- **操作系统**: Linux (Ubuntu 20.04+ / Debian 11+)
- **Python**: 3.11+
- **PLC**: 西门子 S7-1200/1500，DB9数据块
- **硬件**: 工控机 (x86_64)，≥2GB RAM，≥8GB存储

## 快速开始

### 1. 安装

```bash
git clone <repo-url> Ldpj_backend
cd Ldpj_backend
bash scripts/install.sh
source .venv/bin/activate
```

### 2. 开发模式（Mock PLC）

```bash
python main.py --mode mock
```

### 3. 生产模式（真实PLC）

```bash
# 编辑 configs/plc.yaml 设置PLC IP地址
python main.py --mode s7
```

### 4. 模型训练

```bash
python -m train.train_model \
    --data train_data.csv \
    --output models/artifacts/v1.0_20260225 \
    --version v1.0
```

### 5. 模型部署

```bash
bash scripts/deploy_model.sh models/artifacts/v1.0_20260225
```

## 运行时命令

启动后支持以下交互命令：

| 命令 | 功能 |
|------|------|
| `s` | 启动/恢复处理 |
| `e` | 停止/暂停处理 |
| `w` | 切换看门狗 |
| `h` | 执行健康检查 |
| `d` | 打印诊断信息 |
| `q` | 退出程序 |

## API接口

当`ipc.yaml`中`api_server.enabled`为`true`时，系统在配置端口提供HTTP API：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/records` | GET | 查询检测记录（支持时间/舱号/标签过滤） |
| `/records/{id}` | GET | 获取单条记录详情（含原始曲线） |
| `/status` | GET | 系统状态概览 |
| `/health` | GET | 健康检查报告 |

所有接口需在Header中携带 `X-API-Key`。

## 目录结构

```
Ldpj_backend/
├── main.py                 # 主入口
├── requirements.txt        # Python依赖
├── configs/                # 配置文件
│   ├── plc.yaml           # PLC连接与数据映射
│   ├── runtime.yaml       # 运行时参数
│   ├── models.yaml        # 模型管理
│   ├── health.yaml        # 健康检查
│   ├── ipc.yaml           # 外部通讯
│   └── loaders.py         # 配置加载器
├── core/                   # 核心模块
│   ├── polling_engine.py  # PLC高频采集引擎
│   ├── cycle_fsm.py       # 测试周期状态机
│   ├── features.py        # 特征计算
│   ├── logging_setup.py   # 日志初始化
│   └── exceptions.py      # 自定义异常
├── models/                 # 模型管理
│   ├── supervised_xgb.py  # XGBoost推理封装
│   └── artifacts/         # 模型文件存储
├── storage/                # 数据存储
│   ├── database_logger.py # SQLite数据库
│   └── data_exporter.py   # CSV导出
├── health/                 # 健康监控
│   ├── health_checker.py  # 定时健康检查
│   ├── fault_reporter.py  # 故障上报
│   └── fault_codes.py     # 故障码定义
├── integration/            # 外部集成
│   ├── api_server.py      # FastAPI数据服务
│   ├── alarm_pusher.py    # 告警推送
│   └── result_sender.py   # PLC结果写回
├── pipeline/               # 处理流水线
│   ├── processing_loop.py # 主处理循环
│   └── control.py         # 命令控制器
├── train/                  # 训练工具
│   └── train_model.py     # 模型训练脚本
├── scripts/                # 运维脚本
│   ├── install.sh         # 安装脚本
│   └── deploy_model.sh    # 模型部署脚本
└── tests/                  # 单元测试
    ├── conftest.py
    ├── test_features.py
    ├── test_cycle_fsm.py
    ├── test_polling_engine.py
    ├── test_database.py
    └── test_health.py
```

## 模型训练规范

### 特征维度

默认使用7维特征向量（`7d`模式），顺序固定：

| 序号 | 特征名 | 说明 |
|------|--------|------|
| 0 | max | 压力最大值 |
| 1 | min | 压力最小值 |
| 2 | difference | 最大值 - 最小值 |
| 3 | average | 压力均值 |
| 4 | variance | 压力方差 |
| 5 | trend_slope | 线性回归斜率 |
| 6 | cavity_id | 舱室编号 |

### 训练数据格式

CSV文件，必须包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `pressure_data` | string (JSON array) | 压力时间序列 |
| `cavity_id` | int | 舱室编号 (0-5) |
| `label` | int | 标签 (0=漏液, 1=正常) |

### 模型输出

- `xgb_model.json` — XGBoost Booster模型文件
- `xgb_scaler.joblib` — StandardScaler归一化器
- `metadata.json` — 模型元数据（版本、超参数、评估指标）
- `evaluation_report.txt` — 评估报告

## 许可证

内部项目，仅限授权使用。
