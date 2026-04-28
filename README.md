# Ldpj_backend — 边缘 AI 漏液检测后端系统

**版本**: v2.6
**平台**: Linux (Debian/Ubuntu), Python 3.11+, 树莓派 5 / x86 工控机
**PLC**: 西门子 S7-1200/1500, DB_Global [DB9]

---

## 系统简介

Ldpj_backend 部署于产线边缘设备，通过高频轮询西门子 S7 PLC 的传感器数据，
按转盘角度截取整圈压力曲线，用**双轨回归模型**（M1 每舱线性 + M2 全局
XGBoost）实时估计**漏率 Q (Pa·m³/s)**，与产品配置的 `Q_threshold` 比较
判废，并将结果写回 PLC。

### 核心能力

- **高频采集**: 10ms 轮询 25 个舱室；FSM 按 100ms 间隔取样、整圈 70 点
- **角度触发**: RT_Angle 跨 0° 触发，覆盖整周期（baseline / evac / stable / hold / release / baseline）
- **物理输出**: M1 主推理 `Q = β · dp/dt + α`（每舱独立标定）；M2 辅助 43 维 XGBoost 回归 + 一致性交叉校验
- **客户判废**: 按产品 `Q_threshold` 直接出 LEAK / OK；可携带等效缺陷孔径 d (μm)
- **异步写回**: PLC `cabinHealthStatus` 异步队列写回，多舱并发不阻塞采集
- **数据服务**: FastAPI HTTP 接口，外部工控机查询历史/状态/健康
- **告警推送**: 检测到泄漏 / M1-M2 不一致 / 未标定舱时主动推送告警

### 压力标度

```
RT_Pressure: 0 mbar = 常压 (无真空)
            ~600 mbar = 满真空 (正值越大真空度越高)

保压阶段: ~645 mbar, 缓慢下降
  正常密封: dp/dt ≈ 1–3 Pa/s   → Q ≈ 1e-4 Pa·m³/s
  泄漏:     dp/dt ≈ 30–300 Pa/s → Q ≈ 1e-2 Pa·m³/s
```

### 标签定义

| label | 含义 | 触发条件 |
|:---:|---|---|
| 0 | **LEAK** (漏液) | `Q_est > Q_threshold` |
| 1 | **OK** (正常) | `Q_est ≤ Q_threshold` |
| -1 | **N/A** (无判决) | M1 未加载 / Q 低于分辨率 / 无产品阈值 |
| -2 | **NO_BOTTLE** (无瓶) | `hold_max < no_bottle_threshold` |

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/xiaofinesh/Ldpj_backend.git
cd Ldpj_backend
bash scripts/install.sh           # 自动部署最新模型版本到 current/
source .venv/bin/activate
```

### 2. Mock 模式 (开发测试, 无需 PLC)

```bash
python main.py --mode mock
# 系统启动后自动开始采集
```

### 3. S7 模式 (连接真实 PLC)

```bash
# 先编辑 configs/plc.yaml, 设置 PLC IP 地址
python main.py --mode s7
```

### 4. 运行时命令

| 命令 | 功能 |
|:---:|---|
| `s` | 恢复 采集与推理 |
| `e` | 暂停 采集与推理 |
| `x` | 导出数据库到 CSV |
| `w` | 切换看门狗 |
| `m` | 切换显示模式 (正常/调试) |
| `h` | 执行健康检查 |
| `d` | 打印诊断信息 |
| `q` | 退出 |

---

## 数据采集 → 训练 → 部署

v2.6 训练分三步：

### 步骤 1：采集原始数据

```bash
python main.py --mode s7
# 输入 s 开始采集；运行足够时间后输入 e 暂停；输入 x 导出 CSV
```

导出 CSV 包含：`pressure_data`（70 点压力，JSON）、`angle_data`、`features`
（43 维 JSON）、`leak_valve_status`、`cycle_profile_id` 等。

### 步骤 2：计算 Q_measured 标签

```bash
python -m train.prepare_q_data \
    --raw-csv export.csv \
    --cabins-config configs/cabins.yaml \
    --runtime-config configs/runtime.yaml \
    --output train_data.csv
```

按 `Q = V_cabin × |dp/dt|` 给每条记录打 `q_measured` 标签。需要先用
`scripts/calibrate_v_cabin.py` 标定每舱 V_cabin。

### 步骤 3：训练 M1 + M2

```bash
# M1: 每舱线性回归 (主推理)
python -m train.train_m1 \
    --data train_data.csv \
    --output models/artifacts/v2.6.0/m1_coefficients.json \
    --version v2.6.0

# M2: 全局 XGBoost 回归 + top-K 特征选择 (交叉校验)
python -m train.train_m2 \
    --data train_data.csv \
    --output models/artifacts/v2.6.0/ \
    --version v2.6.0 \
    --top-k-features 20
```

### 步骤 4：部署

```bash
bash scripts/deploy_model.sh models/artifacts/v2.6.0
python main.py --mode s7
```

### V_cabin 标定

```bash
python scripts/calibrate_v_cabin.py \
    --cabin 5 \
    --weights-grams 348.2,348.5,348.0 \
    --calibrator alice \
    --notes "first batch"
# CV 超 2% 自动拒写; 历史记入 data/calibration_history/v_cabin_log.csv
```

---

## 机械时序 / 6 段切割

```
转盘: 25 工位, 一圈 ~6900ms (BPH 13000)

  ┌─ angle=0° (trigger): 整圈采集开始, 70 点 @ 100ms ─┐
  │  baseline_pre [0°,57.6°)    瓶进、压紧、常压基线    │
  │  evac         [57.6°,93°)   抽真空段                │
  │  stable       [93°,115°)    稳定段                  │
  │  hold         [115°,273.6°) ★保压检测段 (主信号)    │
  │  release      [273.6°,302.4°) 破真空段              │
  │  baseline_post[302.4°,360°) 瓶出、常压归零          │
  └─ 7 秒后整圈采完 (角度归零兜底) ────────────────────┘
```

`hold` 段是 M1 计算 `trend_slope` 的主段。其它段贡献给 M2 的 43 维特征。

---

## PLC 数据结构

**DB_Global [DB9]**, CabinParam = **20 bytes**:

| 偏移 | 字段 | 类型 | 说明 |
|---:|---|---|---|
| +0 | RT_AI | Int (2B) | 原始模拟量 |
| +2 | RT_Pressure | Real (4B) | 压力值 (0~600 mbar) |
| +6 | RT_Position | Int (2B) | 位号 |
| +8 | RT_Angle | Real (4B) | 角度 (0~360°) |
| +12.0 | leakTestResult_AI | Bool | AI 检测结果 (写回) |
| +12.1 | leakTestResult_PLC | Bool | PLC 检测结果 |
| +14 | **cabinHealthStatus** | Real (4B) | **v2.6: Q_est (Pa·m³/s)** |
| +18.0 | leakValveStatus | Bool | 验证阀状态 (标注用) |

> ⚠️ **HMI 协调要求**：v2.5 时 `cabinHealthStatus` 是概率（0–1）；v2.6
> 改为漏率 Q（典型 1e-7 ~ 1e-2 Pa·m³/s）。**字节格式不变，但 HMI 显示
> 逻辑必须同步切换**，否则上线瞬间用户会看到"健康度从 95% 跌到 0.001"。

Cabin[0] 保留. 系统读取 Cabin[1]~Cabin[25].

---

## 特征工程 (43 维)

70 点压力曲线按角度切成 6 段，每段 7 个统计量 + cavity_id：

```
[
  baseline_pre_max, baseline_pre_min, baseline_pre_difference,
  baseline_pre_average, baseline_pre_variance,
  baseline_pre_trend_slope, baseline_pre_count,

  evac_*  (7),  stable_*  (7),
  hold_*  (7),  ← M1 读 hold_trend_slope
  release_* (7), baseline_post_* (7),

  cavity_id     ← 第 43 维
]
```

`count` 是该段实际包含的点数（短段时仍可识别）。

---

## 推理流水线 (v2.6)

```
PLC --(10ms)--> polling_engine --> CycleFSM --(100ms × 70 点)--> CycleData
                                                                      |
                                                       compute_features_v26
                                                              (43 维)
                                                                      |
                                                            ┌────────┴────────┐
                                                            v                 v
                                              M1 (每舱线性) + M2 (XGBoost 回归)
                                                            \               /
                                                             v             v
                                                       Q_est ← M1 主输出
                                                        |
                                                   product Q_threshold ?
                                                        v
                                              PLC 写回 cabinHealthStatus = Q_est
                                                  (异步队列, 每舱合并最新值)
```

### v2.6 故障码

| 码 | 等级 | 触发条件 |
|---|---|---|
| F010 | WARNING | M1/M2 漏率估计相对差 > 阈值 (默认 20%) |
| F011 | WARNING | M1 表里没有该舱标定（fallback 到均值） |
| F012 | INFO | Q_est 低于系统分辨率 A_resolution（不参与判决） |

完整故障码定义见 `health/fault_codes.py`（F001-F009 来自 v2.5）。

---

## 配置文件

| 文件 | 内容 |
|---|---|
| `configs/plc.yaml` | PLC 连接 + DB9 映射 + 轮询参数 |
| `configs/runtime.yaml` | `cycle_profiles` (周期配方) + `model_inference` + 数据库 |
| `configs/models.yaml` | M1 / M2 / 旧 XGB 路径 |
| `configs/cabins.yaml` | 25 舱 V_cabin 标定值 |
| `configs/products.yaml` | 产品判废参数 (Q_threshold, flow_regime, l_ref_mm) |
| `configs/health.yaml` | 健康检查参数 |
| `configs/ipc.yaml` | API server + 告警推送 |

---

## 目录结构

```
Ldpj_backend/
├── main.py                       # 主入口 (交互命令界面)
├── configs/                      # YAML 配置 (见上表)
├── core/
│   ├── polling_engine.py         # PLC 轮询 (monotonic-tick + seq) + Mock
│   ├── cycle_fsm.py              # 整圈采集 FSM (无累积漂移)
│   ├── cycle_profile.py          # 周期配方抽象 (BPH 档位)
│   ├── curve_segmenter.py        # 按角度切 6 段
│   ├── features.py               # 43 维特征 (闭式 OLS, 优化路径)
│   ├── feature_spec.py           # FEATURE_ORDER_43D 常量
│   ├── q_d_conversion.py         # Q ↔ d (laminar / choked) 双向换算
│   ├── label_spec.py             # 标签 / 时序常量
│   └── exceptions.py             # 自定义异常
├── models/
│   ├── linear_regression_m1.py   # M1: 每舱线性回归 (零依赖)
│   ├── xgb_regressor_m2.py       # M2: 全局 XGBoost 回归
│   ├── supervised_xgb.py         # [DEPRECATED] v2.5 二分类
│   └── artifacts/                # 模型文件 (current/ + archive/)
├── pipeline/
│   └── processing_loop.py        # 主处理循环 (M1+M2 双轨融合)
├── storage/
│   ├── database_logger.py        # SQLite 存储 (zlib BLOB + 流式 export)
│   ├── compression.py            # zlib 压缩工具
│   └── data_exporter.py          # CSV 导出 (单 SELECT, 自动解压)
├── integration/
│   ├── result_sender.py          # PLC 结果异步写回 (per-cabin 合并)
│   ├── api_server.py             # FastAPI 数据服务
│   └── alarm_pusher.py           # HTTP 告警推送
├── health/                       # 健康自检 + 故障码定义
├── train/
│   ├── prepare_q_data.py         # 计算 q_measured
│   ├── train_m1.py               # M1 训练 (含 bootstrap 不确定度)
│   ├── train_m2.py               # M2 训练 (top-K 特征选择)
│   └── train_model.py            # [DEPRECATED] v2.5 二分类训练
├── scripts/
│   ├── install.sh                # 在线安装 + 自动部署最新模型
│   ├── deploy_model.sh           # 模型部署
│   └── calibrate_v_cabin.py      # V_cabin 标定 CLI
├── deploy/
│   └── offline_install.sh        # 离线部署
└── tests/                        # 单元 + 集成测试 (212 项)
```

---

## API 接口

当 `ipc.yaml::api_server.enabled=true` 时：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/records` | GET | 查询记录 (支持时间/舱号/标签过滤) |
| `/records/{id}` | GET | 单条详情 (含解压后压力曲线) |
| `/status` | GET | 系统状态 (M1/M2/PLC 加载情况) |
| `/health` | GET | 健康报告 |

Header: `X-API-Key: <your-key>`

---

## 性能指标 (实测，单舱)

| 阶段 | μs |
|---|---:|
| `compute_features_v26` (43 维) | ~80 |
| `segment_by_angle` (70 点) | ~15 |
| `compress_float_array` (70 点 zlib) | ~13 |
| `M1.predict` | <1 |
| `M2.predict` (XGB, 20 维) | ~50 |
| **单舱完整推理** | **~150 μs** |
| **25 舱并发最坏情况** | **~3.7 ms** |

10ms 轮询周期下还有 6+ ms 的余量。FSM 采用绝对目标 ts 调度，**无累积漂移**。

---

## 测试

```bash
pytest tests/ -v
# 212 passed
```

涵盖：FSM 状态转换 / 特征计算 / 数据库 schema 与压缩 round-trip /
M1+M2 推理路径 / 处理循环集成 / 异步写回 / Q↔d 换算 / 训练脚本端到端。

---

## 许可证

内部项目, 仅限授权使用.
