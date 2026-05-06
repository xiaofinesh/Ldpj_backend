# Ldpj_backend 架构总览（v2.6.2）

> 本文档描述 Ldpj_backend 的端到端数据流、模块职责、关键设计决策与维
> 护契约。重点在"为什么"，对"是什么"只给精炼描述；细节看代码 docstring。

---

## 1. 系统目标

边缘工控机部署的实时漏液检测后端。约束：
- **延迟**：每个完成周期（70 点 / 7s）必须在 10ms 轮询窗口内完成 25 舱推理 + 写回
- **可靠性**：M1 / M2 模型缺失要降级而非崩溃；PLC 通讯抖动要恢复
- **可观测性**：每条推理记录可追溯到原始压力曲线 + 当时的 cycle_profile + 数据质量位
- **兼容性**：v2.5 二分类历史数据要可读；HMI 显示语义切换要硬性确认

---

## 2. 端到端数据流

```
                                    ┌──────────────────────────────┐
                                    │ PLC (S7-1500, DB_Global=DB9) │
                                    └─────────────┬────────────────┘
                                                  │ db_read 26×20=520B
                                                  │ @ 10ms (monotonic-tick)
                                                  ▼
                              ┌───────────────────────────────────┐
                              │ PollingEngine                      │
                              │  - 绝对 next_tick 调度 (无漂移)    │
                              │  - PollFrame.{timestamp,monotonic} │
                              │  - 单调 seq 计数器                 │
                              │  - deque(buffer_size=25000)        │
                              └─────────────┬─────────────────────┘
                                            │ drain_frames_since_seq
                                            ▼
                              ┌─────────────────────────────────┐
                              │ ProcessingLoop._feed_fsm        │
                              │  - 把每帧拆 26 个 CabinFrame    │
                              │  - 分发给对应 CabinFSM          │
                              └─────────────┬───────────────────┘
                                            │
                                            ▼
                ┌────────────── 25 个 CabinFSM (PER CABIN) ──────────────┐
                │  IDLE ──[angle wraps 0°]──► COLLECTING                 │
                │  COLLECTING ──[70 点 OR wrap-back ≥67]──► PROCESSING   │
                │  COLLECTING ──[10s 超时]──► FAULT                      │
                │  采样调度: ts >= next_target_ts (monotonic, += interval) │
                └────────────────────────┬───────────────────────────────┘
                                         │ 状态变 PROCESSING 后, 处理循环取走
                                         ▼
                       ┌────────────────────────────────────┐
                       │ ProcessingLoop._process_cabin      │
                       │  ┌──────────────────────────────┐ │
                       │  │ compute_features_v26          │ │
                       │  │  pressures + angles + cabin   │ │
                       │  │  → 36 维特征 dict             │ │
                       │  └──────────────────────────────┘ │
                       │  ┌──────────────────────────────┐ │
                       │  │ compute_quality_flags         │ │
                       │  │  → 8-bit bitmask              │ │
                       │  └──────────────────────────────┘ │
                       │  ┌──────────────────────────────┐ │
                       │  │ NO_BOTTLE 早退 (hold_max<50) │ │
                       │  └──────────────────────────────┘ │
                       │  ┌──────────────────────────────┐ │
                       │  │ M1.predict(slope, cabin_id)   │ │
                       │  │  Q = β·slope + α              │ │
                       │  │  + bootstrap u_β / u_α        │ │
                       │  └──────────────────────────────┘ │
                       │  ┌──────────────────────────────┐ │
                       │  │ M2.predict(36-dim feature)    │ │
                       │  │  + log10 clamp + scaler       │ │
                       │  └──────────────────────────────┘ │
                       │  ┌──────────────────────────────┐ │
                       │  │ 一致性 / 分辨率 / 阈值判决   │ │
                       │  │  F010 / F012 / LEAK / OK      │ │
                       │  └──────────────────────────────┘ │
                       └─────────┬────────┬────────┬───────┘
                                 │        │        │
                                 ▼        ▼        ▼
                    ┌────────────────┐  ┌────┐  ┌──────────────────┐
                    │ DatabaseLogger │  │PLC │  │ AlarmPusher      │
                    │  zlib BLOB +   │  │写回│  │  LEAK only,      │
                    │  27 列 INSERT  │  │异步│  │  HTTP POST,      │
                    │  全列覆盖      │  │队列│  │  daemon thread   │
                    └────────────────┘  └────┘  └──────────────────┘
                                  ↑                      ↑
                                  │                      │
                              FastAPI                FaultReporter
                              /records, /records/{id} (CRITICAL/ERROR/WARN/INFO)
                              /status, /health
```

### 2.1 关键时序常量

| 常量 | 值 | 来源 | 影响 |
|---|---|---|---|
| 轮询间隔 | 10 ms | `plc.yaml::polling.interval_ms` | 每秒 100 帧/舱 |
| 采集间隔 | 100 ms | `runtime.yaml::cycle_profiles.bph_13000.collection.interval_s` | FSM 取样节拍 |
| 采集点数 | 70 | 同上 `points` | 7s 整圈窗口 |
| 周期总长 | 6900 ms | 同上 `cycle_total_ms` | 单圈生产节拍 (13000 BPH) |
| Wrap-back 下限 | `max(N-3, 0.95N)` | `cycle_fsm._wrap_back_floor` | 70→67 (≥3 点缺失才不放行) |
| 分辨率 A | 1e-5 Pa·m³/s | `runtime.yaml::model_inference.a_resolution` | F012 阈值 |
| 不一致阈值 | 20% | 同上 `m_disagreement_threshold` | F010 阈值 |
| Buffer 大小 | 25000 帧 | `plc.yaml::polling.buffer_size` | ~25s 数据驻留 |

---

## 3. 模块职责

### core/

| 模块 | 行 | 职责 |
|---|---:|---|
| `polling_engine.py` | 470 | PLC 轮询线程 + Mock + S7 包装 + drain_frames_since_seq |
| `cycle_fsm.py` | 285 | 每舱 FSM (4 态) + 绝对目标 ts 调度 + wrap-back floor |
| `cycle_profile.py` | 152 | CycleProfile dataclass + validate + load_active |
| `curve_segmenter.py` | 84 | segment_by_angle + indices (5 段) |
| `features.py` | 230 | compute_features_v26 (36 维) + 闭式 OLS |
| `feature_spec.py` | 73 | FEATURE_ORDER_36D + primary_trend_slope_index |
| `quality_flags.py` | 74 | bitmask 常量 + compute_quality_flags |
| `rate_limit.py` | 58 | warn_throttled 进程级限流 |
| `q_d_conversion.py` | 169 | laminar / choked 双向换算 + C_d clamp warning |
| `label_spec.py` | 53 | LABEL_LEAK / OK / NA / NO_BOTTLE 常量 |
| `exceptions.py` | 32 | 自定义异常类 |

### models/

- `linear_regression_m1.py` — 零 ML 依赖，预测时纯算术，<1μs；fallback 走标定均值
- `xgb_regressor_m2.py` — XGBoost Booster + StandardScaler + metadata；log10 空间 + clamp
- `supervised_xgb.py` — **[DEPRECATED]** v2.5 二分类，未被任何 v2.6 路径引用

### pipeline/

- `processing_loop.py` (470 行) — 整个 v2.6.2 推理大脑：
  - `run_once()` 每 50ms 拉一次 (loop_interval)
  - `_feed_fsm()` 用 seq 游标取增量帧
  - `_process_cabin()` 按 PROCESSING 状态触发；从 features → quality → M1+M2 → 阈值 → 写
  - `_predict_q()` 双轨融合 + F010 / F011 触发
  - `_handle_no_bottle()` NO_BOTTLE 路径只入库不写回 PLC

### integration/

- `result_sender.py` — async writeback + per-cabin coalesce；`PollingEngine.connection` 公共属性而非 `_conn`
- `api_server.py` — FastAPI on port 8000；`/records/{id}` 走 `get_full_record` 避免 BLOB 序列化失败
- `alarm_pusher.py` — daemon thread per alarm；`push_leak_alarm(cabin_id, q_est)` 现在显示 `Q={x:.3e}`

### storage/

- `database_logger.py` — SQLite WAL；27 列 INSERT；`_run_migrations` 精确匹配 "duplicate column" 否则重抛；`iter_records_raw` 行间释放锁
- `compression.py` — zlib level 1，float32 → BLOB；compression ratio ≈ 1.0 但 vs JSON 节省 ~2.5×
- `data_exporter.py` — 单 SELECT，含全量 v2.6 列 + 自动解压 BLOB

### health/

- `fault_reporter.py` — 12 个故障码 (F001–F012)，按 severity 排序的 plc_value
- `health_checker.py` — 7 个周期检查（PLC / 模型 / 磁盘 / 延迟 / 轮询 / FSM / DB）；用 monotonic clock 检 FSM stuck
- `fault_codes.py` — 故障码定义表

### configs/

- `loaders.py` — load_yaml 区分 missing-file (graceful) vs syntax-error (re-raise)；cabins / products / cycle_profile 三个专用加载器

---

## 4. 关键设计决策

### 4.1 为什么选 5 段而不是 6 段？

131808 真实数据的 4 个转折点 mode 是 75°/90°/290°/304°。v2.6 时定义的
"stable" 段（93°–115°）在 112936 真实数据上跟 hold 段斜率比 1.00 ±
0.0000，没有独立的物理瞬态——只是 CAD 时序图上的设计意图，不是测量
得到的相位。删 stable 后 36 维比 43 维抗过拟合更好。

### 4.2 为什么 FSM 用 monotonic 而不是 wall-clock？

NTP 校准会导致 wall-clock 跳变（前跳：所有 ts 突然 >> next_target_ts，
70 点瞬间填满；后跳：ts < next_target_ts 长时间不进，超时 FAULT）。边
缘工控机首次开机 NTP 同步窗口必踩。monotonic clock 不受 NTP 影响，调
度严格基于"开机以来过了多久"。

PollFrame / CabinFrame 同时携带两个 ts：`timestamp`（wall，DB 用）和
`monotonic`（mono，FSM 用）。CabinFSM.update() 优先用 monotonic；测试
fixture 不设置 monotonic 时 fallback 到 timestamp（向后兼容）。

### 4.3 为什么 wrap-back floor 用 max(N-3, 0.95N)？

v2.5 用 70% 固定比例，target=70 时 49 点就放行。在 trigger=0° 附近角
度抖动 0.5°–1° 会让 49 点提前结束，最后丢 21 个点。
新规则是"最多漏 3 个点 **且** 至少 95% 完成"——70 点要求 ≥67，是真·
安全网而非常态。

### 4.4 为什么 M1 是主输出，M2 只做交叉校验？

- M1 物理意义清晰：`Q = β · dp/dt`，β ≈ V_cabin / Δt，可解释、可标定
- M1 训练样本需求小（每舱 ≥ 20 圈）、无过拟合风险、漂移时只改 β
- M2 拟合 43→36 维非线性，捕捉舱体异常 / 释气 / 零点漂移等次级信号
- 两者独立训练，独立加载；任何一个未加载系统都能降级运行
- 一致性差异 > 20% 触发 F010，是数据 / 模型健康的"金丝雀"

### 4.5 为什么写回是异步？

25 舱在同一 crank 位置同时进 PROCESSING 时，25 次同步 RMW 通过
`_io_lock` 顺序写 PLC，每次约 10–20ms（snap7 RTT），合计 250–500ms
阻塞——polling 线程拿不到 `_io_lock`，丢帧。

异步路径：`write_result` 入队 + 唤醒；专用 writer thread 串行处理。
Per-cabin coalesce 保证 25 舱同时入队后实际 PLC 上看到的是每舱最新
verdict；polling 线程从不阻塞。

### 4.6 为什么 quality_flags 在 DB 而不是 features？

特征是给模型看的；quality 是给后续 ML 数据分析看的。塞进 features
会污染模型输入，且 38 维不如 36 维干净。落到独立 INTEGER 列让分析
端 `WHERE quality_flags & 0x10 != 0` 直接筛"hold 段降级"行。

### 4.7 为什么 `loaders.load_yaml` 只 catch 文件缺失？

v2.6.1 之前 `load_yaml` 静默吞所有异常返回 `{}`。一个 yaml 笔误（多
冒号、缩进错）就让所有 config 变空，下游全用默认值，启动后系统看
似正常但行为完全不对。
v2.6.2 起：missing-file 返回 `{}`（开发环境优雅降级），syntax-error
和 IO error 直接抛——硬性失败比"假装能跑"安全。

### 4.8 为什么 PLC `cabinHealthStatus` 语义变更要做强制确认？

字节格式没变（4 字节 REAL），但语义从 [0, 1] 概率变成 1e-7 ~ 1e-2
漏率。HMI 直接显示这个 REAL 值，切换瞬间用户看到"健康度从 95% 跌到
0.001"会以为系统崩溃，触发产线停机或现场恐慌。
`install.sh` / `deploy_model.sh` 在 cp 进 `current/` 之前强制 `read`
确认，CI 用 `LDPJ_SKIP_HMI_CONFIRM=1` 跳过。

---

## 5. 数据库 schema（关键列）

```sql
CREATE TABLE test_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- v2.5 base
    batch_id TEXT, cavity_id INTEGER, timestamp TEXT, label INTEGER,
    point_count INTEGER, duration_s REAL,
    leak_valve_status INTEGER, end_angle REAL,
    -- v2.5 legacy text columns (placeholder "" / NULL in v2.6)
    pressure_data TEXT NOT NULL, angle_data TEXT,
    ai_data TEXT, position_data TEXT,
    features TEXT,                       -- 36-key JSON
    probability REAL,                    -- 兼容: v2.6 写 q_est
    confidence REAL,                     -- 兼容: v2.6 写 1 - rel_unc
    model_version TEXT,
    -- v2.6 columns
    cycle_profile_id TEXT,
    pressure_data_compressed BLOB,       -- zlib float32 (~280B)
    angle_data_compressed BLOB,          -- zlib float32 (~280B)
    q_est REAL, q_threshold REAL, q_uncertainty REAL,
    m1_q REAL, m2_q REAL, m_disagreement REAL,
    product_id TEXT,
    quality_flags INTEGER DEFAULT 0,     -- bitmask, see core/quality_flags.py
    created_at TEXT
);
```

迁移规则：`_run_migrations` 用 idempotent ALTER；只匹配 `"duplicate
column name"` pass，其它 OperationalError 重抛（避免 v2.6.1 时静默
吞磁盘满 / 语法错的问题）。

---

## 6. 训练流水线

```
原始 CSV (导出)
   │
   ▼
prepare_q_data.py     ← 用 cabins.yaml 的 V_cabin 给每条记录算 q_measured
   │
   ▼
train_data.csv (有 q_measured + dp_dt 列)
   │
   ├──► train_m1.py    ← 每舱独立线性回归
   │     │  ┌─ np.polyfit + 1000-resample bootstrap → β, α, u_β, u_α
   │     │  ├─ R² < min_r2 (默认 0.99) → 该舱被 SKIP
   │     │  └─ 输出: cabin_coefs[cabin_id] = {β, α, u_β, u_α, r², n}
   │     ▼
   │  m1_coefficients.json
   │
   └──► train_m2.py    ← 全局 XGBoost 回归 + 特征选择
         │  ┌─ Pass 1: 全 36 维训, gain 排序
         │  ├─ Pass 2: top-K 重训 + 新 StandardScaler
         │  ├─ 目标: log10(Q) clip [1e-7, 1.0]
         │  └─ 训练/测试 split 按 round_id (避免 leakage)
         ▼
      m2_xgb_model.json + m2_xgb_scaler.joblib + m2_metadata.json
```

V_cabin 标定（首次或定期）：

```bash
scripts/calibrate_v_cabin.py --cabin <N> --weights-grams w1,w2,w3,...
# CV ≤ 2% → 写 cabins.yaml + 历史 CSV (accepted=yes)
# CV > 2% → 拒写 yaml, 仍记历史 (accepted=no, 标 REJECTED)
```

部署：

```bash
scripts/deploy_model.sh models/artifacts/v2.6.2
# 强制 HMI 确认 → 归档 current/ 到 archive/<old_version>_<ts>/
# → cp 新 artifact 到 current/ → 更新 models.yaml
```

---

## 7. 可观测性

### 7.1 日志通道

- 文件日志（rotated, 5MB × 5 backup, INFO+）：`ldpj_backend.log`
- 控制台（normal mode 仅 WARNING+，debug mode 全量）
- StatusReporter 每 30s 心跳镜像到文件 logger
- rate-limited warning：`core/rate_limit.warn_throttled` 防止 stuck
  cabin 把日志刷爆（首次必出，之后每 100 次或 60s 一次）

### 7.2 健康检查

`HealthChecker` 每 60s 跑全套 7 项检查，结果按 severity 转换为 F001–
F009 故障码（active 与否由 FaultReporter 维护）。`/health` API 实时
触发完整检查并返回 JSON 报告。

### 7.3 数据质量

- `quality_flags` 列：每条记录的 8-bit bitmask（短段 / clamped C_d / etc.）
- `m_disagreement` 列：M1/M2 相对差，超 20% 触 F010
- `q_uncertainty` 列：M1 的 1-σ 不确定度（β/α 不确定度合成）
- 训练时用 `passes_acceptance` 决定写不写入 m1_coefficients.json，未达
  标的舱在生产中走 `_predict_fallback` + F011

---

## 8. 与 v2.5 / v2.6 / v2.6.1 的兼容点

| 维度 | v2.5 | v2.6 | v2.6.1 | v2.6.2 |
|---|---|---|---|---|
| 段数 | n/a (保压 15 点) | 6 (含 stable) | **5** | 5 |
| 特征维度 | 7 | 43 | **36** | 36 |
| 推理输出 | label + probability | Q_est | Q_est | Q_est |
| PLC `cabinHealthStatus` | probability ∈ [0,1] | **Q_est (Pa·m³/s)** | Q_est | Q_est |
| 训练目标 | 二分类 logloss | log10(Q) regression | log10(Q) regression | log10(Q) regression |
| FSM 时钟 | wall-clock | wall-clock | wall-clock | **monotonic** |
| Wrap-back floor | n/a (固定点数) | 70% target | 70% target | **max(N-3, 0.95N)** |
| DB schema | 17 列 | 26 列 | 27 列 (+quality_flags) | 27 列 |

迁移注意：
- v2.5 的 m1_coefficients.json **不存在**（M1 是 v2.6 引入），无需迁移
- v2.5 的 supervised_xgb.json 在 v2.6+ 无人引用，可随时删（保留作 audit）
- v2.6 训过的 M2 与 v2.6.1+ **不兼容**（feature_subset 引用了 stable_*
  特征），需要重训；M1 系数则继续可用（只看 hold_trend_slope）
- v2.6 的 `quality_flags` bit 3 是 `QF_SHORT_STABLE`；v2.6.1+ 重排为
  `QF_SHORT_HOLD`。读老库需按 `created_at` 区分语义

---

## 9. 维护契约

### 9.1 新增配置项

1. 在对应 yaml 加默认值
2. 在 `configs/loaders.py` 加专用 loader（如果是新文件）或直接 .get
3. 在使用方 `__init__` 用 `cfg.get(key, default)` 读，不要在 hot path 读
4. 在 README 配置文件章节加一行
5. 在测试里加 fixture 验证默认值不变

### 9.2 新增故障码

1. 在 `health/fault_codes.py::FAULT_CODES` 加条目（F0XX, level, plc_value）
2. 在触发处调 `fault_reporter.raise_fault(code, msg)` + `resolve_fault(code)`（如适用）
3. 在 README "故障码" 表加一行
4. 测试 raise / resolve 的双向能正常切换

### 9.3 新增段（cycle_profile.SECTION_NAMES）

需协调改动以下文件：
- `core/cycle_profile.py::SECTION_NAMES`
- `configs/runtime.yaml::cycle_profiles.<id>.sections`
- `core/quality_flags.py::_SECTION_BITS`（如果要为新段加位）
- `core/feature_spec.py` 自动推导（无需改）
- 所有测试 fixture 的 `sections={...}` 字典
- M2 的 metadata.json::feature_subset 重新训练（旧模型不兼容）
- README "机械时序" + "特征工程" 章节

### 9.4 新增 v2.x 版本

1. 改 `main.py:_print_banner` 字符串
2. 改 `main.py` 模块 docstring
3. 改 `integration/api_server.py` FastAPI version
4. 改 `configs/runtime.yaml` 头注释
5. 改 README 顶部版本字段 + 版本历史章节
6. 在本文件第 8 节"兼容点"表加一列
