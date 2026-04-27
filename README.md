# Ldpj_backend — 边缘AI漏液检测后端系统

**版本**: v2.5  
**平台**: Linux (Debian/Ubuntu), Python 3.11+, 树莓派5 / x86工控机  
**PLC**: 西门子 S7-1200/1500, DB_Global [DB9]

---

## 系统简介

Ldpj_backend 部署于产线边缘设备，通过高频轮询西门子S7 PLC的传感器数据，在旋转检测转盘的保压阶段截取压力曲线，利用 XGBoost 模型实时判断瓶体是否存在泄漏，并将结果写回PLC。

### 核心能力

- **实时采集**: 10ms 高频轮询 25 个舱室的压力/角度数据 (FSM 按 200ms 间隔取样)
- **角度触发**: 按 RT_Angle 精确截取保压阶段曲线 (100°~276°)
- **AI推理**: XGBoost 7维特征分类, 单次推理 <10ms
- **验证阀标注**: 采集阶段通过 leakValveStatus 自动标注训练数据
- **数据服务**: FastAPI HTTP 接口, 支持外部工控机查询
- **告警推送**: 检测到泄漏时主动推送 HTTP 告警

### 压力标度

```
RT_Pressure: 0 mbar = 常压 (无真空)
            ~600 mbar = 满真空 (正值越大真空度越高)

保压阶段: ~645 mbar, 缓慢下降 10~50 mbar
  正常密封: 下降 5~15 mbar (曲线平缓)
  泄漏:     下降 80~120 mbar (曲线陡峭)
```

### 标签定义

| label | 含义 | 触发条件 |
|:---:|---|---|
| 0 | **LEAK** (漏液) | 模型判定 / leakValveStatus=true |
| 1 | **OK** (正常) | 模型判定 / leakValveStatus=false |
| -1 | **N/A** (无推理) | 模型未加载, 采集阶段 |

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/xiaofinesh/Ldpj_backend.git
cd Ldpj_backend
bash scripts/install.sh
source .venv/bin/activate
```

### 2. Mock 模式 (开发测试, 无需PLC)

```bash
python main.py --mode mock
# 系统启动后处于暂停状态, 输入 s 开始采集
```

### 3. S7 模式 (连接真实PLC)

```bash
# 先编辑 configs/plc.yaml, 设置 PLC IP 地址
python main.py --mode s7
# 输入 s 开始采集
```

### 4. 运行时命令

系统启动后显示交互菜单, 两种模式操作完全一致:

| 命令 | 功能 |
|:---:|---|
| `s` | 开始/恢复采集 |
| `e` | 暂停采集 |
| `w` | 切换看门狗 |
| `h` | 健康检查报告 |
| `d` | 诊断信息 |
| `x` | 导出数据库到CSV |
| `q` | 退出 |

---

## 数据采集与训练流程

### 步骤1: 采集原始数据

```bash
python main.py --mode s7   # 或 --mode mock
# 输入 s 开始采集
# 运行足够时间后, 输入 e 暂停
# 输入 x 导出CSV
```

导出的CSV中:
- `label=-1`: 模型未加载, 无推理
- `leak_valve_statuses`: 验证阀状态序列 (JSON), 用于训练标注
- `pressure_data`: 保压阶段压力曲线 (JSON)

### 步骤2: 训练模型

```bash
# 使用新系统采集的数据 (leakValveStatus 自动标注):
python -m train.train_model \
    --data export_20260316.csv \
    --output models/artifacts/v1.0 \
    --version v1.0 \
    --min-pressure 100

# 使用旧系统CSV数据 (标签需翻转):
python -m train.train_model \
    --data old_data.csv \
    --output models/artifacts/v1.0 \
    --flip-labels \
    --label-column prediction \
    --min-pressure -5
```

### 步骤3: 部署模型

```bash
bash scripts/deploy_model.sh models/artifacts/v1.0
python main.py --mode s7
```

---

## 机械时序

```
转盘: 25工位, 一圈 ~6944ms (~277.8ms/工位)

  Pos 0─5:   空闲 (瓶子进入)
  Pos 5:     开始抽真空 (400ms)
  Pos ~6.5:  真空建立 (~645 mbar)
  ┌─── Pos 6.9 (angle=100°): AI采集开始 ───┐
  │    保压阶段: ~645 缓慢下降              │ ≈3.6s
  │    正常: 下降 5~15 mbar                 │ ~34 点
  │    泄漏: 下降 80~120 mbar               │ @100ms
  └─── Pos 19.2 (angle=276°): AI采集结束 ──┘
  Pos 21:    复压开始
  Pos 24:    结果输出
```

---

## PLC 数据结构

**DB_Global [DB9]**, CabinParam = **20 bytes**:

| 偏移 | 字段 | 类型 | 说明 |
|---:|---|---|---|
| +0 | RT_AI | Int (2B) | 原始模拟量 |
| +2 | RT_Pressure | Real (4B) | 压力值 (0~600 mbar) |
| +6 | RT_Position | Int (2B) | 位号 |
| +8 | RT_Angle | Real (4B) | 角度 (0~360°) |
| +12.0 | leakTestResult_AI | Bool | AI检测结果 (写回) |
| +12.1 | leakTestResult_PLC | Bool | PLC检测结果 |
| +14 | cabinHealthStatus | Real (4B) | 健康度 |
| +18.0 | leakValveStatus | Bool | 验证阀状态 (标注用) |

Cabin[0] 保留无实际意义. 系统读取 Cabin[1]~Cabin[25] (start_offset=20).

---

## 特征工程

7维特征向量, 基于保压阶段压力曲线计算:

| # | 特征 | 计算方式 |
|---|---|---|
| 0 | max | 压力最大值 |
| 1 | min | 压力最小值 |
| 2 | difference | max - min |
| 3 | average | 均值 |
| 4 | variance | 方差 |
| 5 | trend_slope | 线性回归斜率 |
| 6 | cavity_id | 舱室编号 |

---

## 目录结构

```
Ldpj_backend/
├── main.py                     # 主入口 (交互命令界面)
├── configs/
│   ├── plc.yaml                # PLC连接 & DB9映射
│   ├── runtime.yaml            # 运行时参数 (角度触发/标签/阈值)
│   ├── models.yaml             # 模型路径管理
│   ├── health.yaml             # 健康检查配置
│   └── ipc.yaml                # API/告警推送配置
├── core/
│   ├── polling_engine.py       # PLC轮询引擎 + Mock模拟器
│   ├── cycle_fsm.py            # 角度触发状态机
│   ├── features.py             # 7维特征计算
│   ├── label_spec.py           # 标签/时序常量定义
│   └── exceptions.py           # 自定义异常
├── models/
│   ├── supervised_xgb.py       # XGBoost推理封装
│   └── artifacts/              # 模型文件 (current/ + archive/)
├── pipeline/
│   ├── processing_loop.py      # 主处理循环
│   └── control.py              # 交互命令控制器
├── storage/
│   ├── database_logger.py      # SQLite存储
│   └── data_exporter.py        # CSV导出
├── integration/
│   ├── result_sender.py        # PLC结果写回
│   ├── api_server.py           # FastAPI数据服务
│   └── alarm_pusher.py         # HTTP告警推送
├── health/                     # 健康自检模块
├── train/
│   └── train_model.py          # 模型训练脚本
├── scripts/
│   ├── install.sh              # 在线安装
│   └── deploy_model.sh         # 模型部署
├── deploy/
│   └── offline_install.sh      # 离线部署
└── tests/                      # 单元测试
```

---

## API 接口

当 `ipc.yaml` 中 `api_server.enabled=true` 时:

| 端点 | 方法 | 说明 |
|---|---|---|
| `/records` | GET | 查询记录 (支持时间/舱号/标签过滤) |
| `/records/{id}` | GET | 单条详情 (含原始曲线) |
| `/status` | GET | 系统状态 |
| `/health` | GET | 健康报告 |

Header: `X-API-Key: <your-key>`

---

## 许可证

内部项目, 仅限授权使用.
