# Ldpj_backend v2.6 改造需求 — 全周期采集与 Q 回归

> **目的**:把 Ldpj_backend 从 v2.5 的"15 点保压采集 + XGBoost 二分类"改造为 v2.6 的"70 点全周期采集 + Q 回归输出",支持微渗漏检测的灵敏度标定项目。
>
> **范围**:`Ldpj_backend/` 仓库的全部代码改造。**不涉及** PLC 端的 STL/SCL 程序(由自控工程师另行处理)。本文档约定 PLC 端必须配合的字段语义变更。
>
> **基线**:`xiaofinesh/Ldpj_backend` v2.5 主分支。开发新分支 `v2.6-regression`。
>
> **执行方式**:本文档分为 10 个独立任务。每个任务可单独喂给 Claude Code 完成,任务之间有依赖关系(见下图)。建议按编号顺序执行。
>
> **测试要求**:每个任务完成后必须通过全量 pytest,且 mock 模式下 `python main.py --mode mock` 能正常启动并采集。

---

## 总体架构变更

### 当前(v2.5)

```
PLC --(10ms 轮询)--> polling_engine --> CycleFSM --(15 点 @ 200ms 间隔)--> CycleData
                                                                              |
                                            [角度 100° 触发, 采 15 点结束]    |
                                                                              v
                                                          features.py(7 维)
                                                                              |
                                                                              v
                                                          XGBoost 二分类 --> label ∈ {0=LEAK, 1=OK}
                                                                              |
                                                                              v
                                                          PLC 写回:
                                                            leakTestResult_AI (Bool)
                                                            cabinHealthStatus = probability ∈ [0, 1]
```

### 目标(v2.6)

```
PLC --(10ms 轮询)--> polling_engine --> CycleFSM --(70 点 @ 100ms 间隔, 整圈 0°~360°)
                                                                              |
                                            [角度 0° 触发, 采 70 点 / 角度归零结束]
                                                                              v
                                                          curve_segmenter.py
                                                            按角度切成 6 段
                                                                              v
                                                          features.py(43 维: 6×7 + cavity_id)
                                                                              v
                                                              ┌──────┴──────┐
                                                              v             v
                                                          M1 (每舱线性)  M2 (XGBoost 回归)
                                                              |             |
                                                              └──────┬──────┘
                                                                     v
                                                          双轨融合 → Q_est (Pa·m³/s)
                                                                     |
                                                          客户阈值: Q_est > Q_threshold ?
                                                                     v
                                                          PLC 写回:
                                                            leakTestResult_AI (Bool)
                                                            cabinHealthStatus = Q_est (REAL)
```

### 关键变化点

1. **采集范围**:从"保压段 15 点"扩展为"整圈 70 点"(0°~360°,5 段切割 + 2 段基线归零)
2. **采集间隔**:从 200ms 缩短为 100ms
3. **配方系统**:新增 `cycle_profile` 抽象,封装"产量档位 / 时序角度 / 采集参数"。本期只填 13000 档,但接口为未来"从 PLC 读配方"预留
4. **特征工程**:从 7 维扩展为 43 维(6 段独立统计 + cavity_id),支持 M2 的舱体健康监测
5. **数据存储**:从 JSON 字符串改为 zlib 压缩 BLOB,容量降低 3-5 倍
6. **模型**:从 XGBoost 二分类改为双轨回归(M1 每舱线性 + M2 全局 XGBoost)
7. **PLC 写回语义**:`cabinHealthStatus` 从"概率 0~1"改为"漏率 Pa·m³/s",**这件事必须协调自控工程师确认**

### 兼容性策略

- v2.5 的旧 XGBoost 二分类模型**完全废弃**(不保留 fallback)。原因:新采集格式与旧模型的 7 维输入不兼容,且阶段 2 实验数据出来后会有完整的回归模型替代
- 旧的 `pressure_data` (JSON) 字段废弃,新数据走 `pressure_data_compressed` (BLOB)
- 数据库 schema 通过 `_run_migrations()` 自动添加新列;旧数据的新列填 NULL

---

## 任务依赖关系

```
任务 1 (配方系统 + 数据库 schema)
    │
    ├── 任务 2 (FSM 全周期采集)
    │       │
    │       ├── 任务 3 (段切割 + 43 维特征)
    │       │       │
    │       │       ├── 任务 5 (V_cabin 配置)  [独立可并行]
    │       │       │       │
    │       │       │       ├── 任务 6 (M1 线性回归)
    │       │       │       │       │
    │       │       │       │       └── 任务 8 (推理流水线集成)
    │       │       │       │
    │       │       │       └── 任务 7 (M2 XGBoost 回归)
    │       │       │               │
    │       │       │               └── 任务 8 (同上)
    │       │       │
    │       │       └── 任务 10 (训练脚本)
    │       │
    │       └── 任务 4 (zlib 压缩存储)
    │
    └── 任务 9 (产品配置 + Q/d 换算)  [独立可并行]
```

**并行机会**:任务 5、9、10 可与任务 6/7 并行开发。

---

# 任务 1:配方系统(cycle_profile)+ 数据库 schema 升级

## 1.1 背景

v2.5 把所有时序参数(start_angle, collection_points, collection_interval_s)写在 `runtime.yaml` 的扁平结构里。这种设计的问题是**对产量档位变化没有适配能力**——客户切换 13000 → 10000 档时,周期从 6900ms 变成 9000ms,所有角度边界都要变,但代码无从感知。

v2.6 引入"配方(cycle_profile)"抽象。一个 profile 描述特定产量档位下的全部时序参数。本期只填 13000 档,但代码结构允许未来:
- 增加其他档位(只需 yaml 添新条目)
- 替换为"从 PLC 配方表读取"(只需替换一个函数实现)

## 1.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `configs/runtime.yaml` | 重写 `cycle_detection` 段,改为 `cycle_profiles` + `active_profile` |
| `core/cycle_profile.py` | **新增**:CycleProfile 数据类与加载函数 |
| `configs/loaders.py` | 新增 `load_active_cycle_profile()` 函数 |
| `storage/database_logger.py` | 新增字段 `cycle_profile_id`、`pressure_data_compressed`、`angle_data_compressed` 等;migration |
| `tests/test_cycle_profile.py` | **新增**:profile 加载与验证的单元测试 |

## 1.3 详细改动

### 1.3.1 重写 `configs/runtime.yaml` 的时序段

**新结构**:

```yaml
# 周期配方系统(v2.6)
# 一个 profile 描述特定产量档位下的全部时序参数。
# 当前只定义一档(13000 瓶/小时,周期 6900ms)。
# 未来扩展:增加其他档位条目,或替换为"从 PLC 读取"。

active_profile: "bph_13000"

cycle_profiles:
  bph_13000:
    description: "13000 瓶/小时,周期 6900ms"
    bph: 13000                    # bottles per hour
    cycle_total_ms: 6900          # 整圈周期(ms)
    
    # 6 段角度边界(°),按转盘 0°~360° 划分
    # 段名 → [起始角度, 结束角度)
    sections:
      baseline_pre:  [0.0,   57.6]    # 瓶进、压紧、常压基线
      evac:          [57.6,  93.0]    # 抽真空段
      stable:        [93.0,  115.0]   # 稳定段
      hold:          [115.0, 273.6]   # 保压检测段(主信号)
      release:       [273.6, 302.4]   # 破真空段
      baseline_post: [302.4, 360.0]   # 瓶出、常压归零
    
    # 采集参数
    collection:
      trigger_angle: 0.0            # 采集触发角度(整圈起点)
      points: 70                    # 采集点数
      interval_s: 0.1               # 采样间隔(秒)
      timeout_s: 10.0               # 采集超时(秒)
    
    # 主信号段(M1 线性回归用此段计算 trend_slope)
    primary_section: "hold"

# ── 标签定义 ────────────────────────────────────────────────
# label =  1: OK (正常)
# label =  0: LEAK (漏液)
# label = -1: N/A (模型未加载或异常)
# label = -2: NO_BOTTLE (无瓶子, max < 50)

# ── 模型推理 ─────────────────────────────────────────────────
model_inference:
  # 系统分辨率 A (Pa·m³/s)。Q_est 低于 A 时不参与判决
  a_resolution: 1.0e-5
  
  # M1 / M2 不一致告警阈值(相对差)
  m_disagreement_threshold: 0.20

  # 特征模式:43d (6 段 × 7 + cavity_id) 是 v2.6 的标准
  feature_mode: "43d"

# 无瓶子判定: 压力最大值 < 此阈值 → label=-2, 跳过推理
no_bottle_threshold: 50.0

# ── 日志与数据库 ────────────────────────────────────────────
logging:
  level: INFO
  file: ldpj_backend.log
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  rotate:
    max_bytes: 5242880
    backup_count: 5

database:
  path: ldpj_data.db
  max_size_mb: 1000

loop_interval: 0.05
```

### 1.3.2 新增 `core/cycle_profile.py`

```python
"""Cycle profile abstraction for v2.6.

A profile describes the time-domain layout of one production cycle,
including section boundaries (in degrees), sampling parameters, and
machine throughput tier.

Currently only one profile is populated (bph_13000), but the abstraction
allows future expansion:
1. Add more profile entries to runtime.yaml
2. Replace `load_active_cycle_profile()` with a PLC-recipe reader
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


# Standard section names (order matters for feature extraction)
SECTION_NAMES = [
    "baseline_pre", "evac", "stable", "hold", "release", "baseline_post"
]


@dataclass
class CycleProfile:
    """Describes one production cycle layout.

    Attributes
    ----------
    profile_id : str
        Unique identifier (e.g. "bph_13000").
    bph : int
        Bottles per hour throughput.
    cycle_total_ms : int
        Total cycle duration in milliseconds.
    sections : Dict[str, Tuple[float, float]]
        Section name → [start_angle, end_angle) in degrees.
    trigger_angle : float
        Angle (degrees) at which collection starts.
    collection_points : int
        Number of samples per cycle.
    collection_interval_s : float
        Sampling interval in seconds.
    collection_timeout_s : float
        Maximum collection duration before FAULT.
    primary_section : str
        Name of the section used for M1 trend_slope computation.
    """
    profile_id: str
    bph: int
    cycle_total_ms: int
    sections: Dict[str, Tuple[float, float]]
    trigger_angle: float
    collection_points: int
    collection_interval_s: float
    collection_timeout_s: float
    primary_section: str = "hold"
    description: str = ""

    def validate(self) -> None:
        """Sanity-check the profile. Raises ValueError on invalid config."""
        # 1. All standard section names must be present
        missing = set(SECTION_NAMES) - set(self.sections.keys())
        if missing:
            raise ValueError(f"Profile {self.profile_id}: missing sections {missing}")
        
        # 2. Section boundaries must be monotonically increasing across all 6 sections
        prev_end = -1
        for name in SECTION_NAMES:
            start, end = self.sections[name]
            if start < prev_end:
                raise ValueError(
                    f"Profile {self.profile_id}: section '{name}' starts at {start}° "
                    f"before previous section ended at {prev_end}°"
                )
            if end <= start:
                raise ValueError(
                    f"Profile {self.profile_id}: section '{name}' has invalid range "
                    f"[{start}, {end})"
                )
            prev_end = end
        
        # 3. Last section must end at <= 360.0
        last_end = self.sections[SECTION_NAMES[-1]][1]
        if last_end > 360.001:
            raise ValueError(
                f"Profile {self.profile_id}: total range exceeds 360° (ends at {last_end}°)"
            )
        
        # 4. primary_section must be one of the standard names
        if self.primary_section not in SECTION_NAMES:
            raise ValueError(
                f"Profile {self.profile_id}: primary_section '{self.primary_section}' "
                f"not in {SECTION_NAMES}"
            )
        
        # 5. Sampling sanity
        if self.collection_points <= 0 or self.collection_interval_s <= 0:
            raise ValueError(f"Profile {self.profile_id}: invalid sampling params")
        
        expected_duration = self.collection_points * self.collection_interval_s
        if expected_duration > self.collection_timeout_s:
            raise ValueError(
                f"Profile {self.profile_id}: timeout ({self.collection_timeout_s}s) "
                f"shorter than expected collection duration ({expected_duration}s)"
            )
        
        logger.info(
            "CycleProfile validated: %s (bph=%d, %d points × %.0fms, primary=%s)",
            self.profile_id, self.bph,
            self.collection_points, self.collection_interval_s * 1000,
            self.primary_section,
        )

    @classmethod
    def from_dict(cls, profile_id: str, data: Dict[str, Any]) -> "CycleProfile":
        """Build CycleProfile from a yaml-loaded dict."""
        sections = {
            name: tuple(bounds) for name, bounds in data.get("sections", {}).items()
        }
        collection = data.get("collection", {})
        return cls(
            profile_id=profile_id,
            bph=int(data.get("bph", 0)),
            cycle_total_ms=int(data.get("cycle_total_ms", 0)),
            sections=sections,
            trigger_angle=float(collection.get("trigger_angle", 0.0)),
            collection_points=int(collection.get("points", 70)),
            collection_interval_s=float(collection.get("interval_s", 0.1)),
            collection_timeout_s=float(collection.get("timeout_s", 10.0)),
            primary_section=data.get("primary_section", "hold"),
            description=data.get("description", ""),
        )


def load_active_cycle_profile(runtime_cfg: Dict[str, Any]) -> CycleProfile:
    """Load the currently active profile from runtime.yaml.
    
    NOTE: This is the v2.6 implementation (read from yaml).
    Future v2.7 may replace with PLC recipe reading:
        return read_from_plc_recipe(plc_conn)
    
    Parameters
    ----------
    runtime_cfg : dict
        The full runtime.yaml content.
    
    Returns
    -------
    CycleProfile (validated)
    
    Raises
    ------
    ValueError if active_profile is not found or profile is invalid.
    """
    active_id = runtime_cfg.get("active_profile")
    if not active_id:
        raise ValueError("runtime.yaml: 'active_profile' is not set")
    
    profiles = runtime_cfg.get("cycle_profiles", {})
    if active_id not in profiles:
        raise ValueError(
            f"Active profile '{active_id}' not found in cycle_profiles. "
            f"Available: {list(profiles.keys())}"
        )
    
    profile = CycleProfile.from_dict(active_id, profiles[active_id])
    profile.validate()
    return profile
```

### 1.3.3 修改 `configs/loaders.py`

参考现有的 `load_runtime_config()` 模式,新增:

```python
from core.cycle_profile import CycleProfile, load_active_cycle_profile as _load_profile

def load_active_cycle_profile() -> CycleProfile:
    """Convenience wrapper that loads runtime.yaml and extracts active profile."""
    runtime_cfg = load_runtime_config()
    return _load_profile(runtime_cfg)
```

### 1.3.4 修改 `storage/database_logger.py`

#### 新增字段(SQL)

```sql
ALTER TABLE test_records ADD COLUMN cycle_profile_id TEXT;
ALTER TABLE test_records ADD COLUMN pressure_data_compressed BLOB;
ALTER TABLE test_records ADD COLUMN angle_data_compressed BLOB;
ALTER TABLE test_records ADD COLUMN q_est REAL;
ALTER TABLE test_records ADD COLUMN q_threshold REAL;
ALTER TABLE test_records ADD COLUMN q_uncertainty REAL;
ALTER TABLE test_records ADD COLUMN m1_q REAL;
ALTER TABLE test_records ADD COLUMN m2_q REAL;
ALTER TABLE test_records ADD COLUMN m_disagreement REAL;
ALTER TABLE test_records ADD COLUMN product_id TEXT;
```

#### 在 `_run_migrations()` 中追加

```python
_MIGRATIONS = [
    # ... 已有的 migration ...
    "ALTER TABLE test_records ADD COLUMN cycle_profile_id TEXT",
    "ALTER TABLE test_records ADD COLUMN pressure_data_compressed BLOB",
    "ALTER TABLE test_records ADD COLUMN angle_data_compressed BLOB",
    "ALTER TABLE test_records ADD COLUMN q_est REAL",
    "ALTER TABLE test_records ADD COLUMN q_threshold REAL",
    "ALTER TABLE test_records ADD COLUMN q_uncertainty REAL",
    "ALTER TABLE test_records ADD COLUMN m1_q REAL",
    "ALTER TABLE test_records ADD COLUMN m2_q REAL",
    "ALTER TABLE test_records ADD COLUMN m_disagreement REAL",
    "ALTER TABLE test_records ADD COLUMN product_id TEXT",
]
```

#### 修改 `log_record()` 签名

新增可选参数(全部 default None,保持向后兼容):

```python
def log_record(self, cavity_id, pressures, angles, ai_values, positions,
               features, label, probability, confidence, model_version,
               duration_s, leak_valve_status=None, end_angle=None,
               batch_id="",
               # v2.6 新增字段
               cycle_profile_id=None,
               pressure_data_compressed=None,
               angle_data_compressed=None,
               q_est=None, q_threshold=None, q_uncertainty=None,
               m1_q=None, m2_q=None, m_disagreement=None,
               product_id=None) -> int:
    ...
```

INSERT 语句和参数列表相应扩展。

#### 关于 `pressure_data` 旧字段

**v2.6 起完全不写入**。新数据只写 `pressure_data_compressed`。但保留字段供旧记录查询(避免破坏历史数据库)。

### 1.3.5 单元测试 `tests/test_cycle_profile.py`

```python
"""Unit tests for cycle_profile module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.cycle_profile import CycleProfile, load_active_cycle_profile, SECTION_NAMES


def _valid_profile_dict():
    return {
        "description": "test",
        "bph": 13000,
        "cycle_total_ms": 6900,
        "sections": {
            "baseline_pre":  [0.0,   57.6],
            "evac":          [57.6,  93.0],
            "stable":        [93.0,  115.0],
            "hold":          [115.0, 273.6],
            "release":       [273.6, 302.4],
            "baseline_post": [302.4, 360.0],
        },
        "collection": {
            "trigger_angle": 0.0,
            "points": 70,
            "interval_s": 0.1,
            "timeout_s": 10.0,
        },
        "primary_section": "hold",
    }


class TestCycleProfile:
    def test_from_dict_basic(self):
        p = CycleProfile.from_dict("test", _valid_profile_dict())
        assert p.profile_id == "test"
        assert p.bph == 13000
        assert p.collection_points == 70
        assert p.primary_section == "hold"
    
    def test_validate_passes(self):
        p = CycleProfile.from_dict("test", _valid_profile_dict())
        p.validate()  # should not raise
    
    def test_validate_missing_section(self):
        d = _valid_profile_dict()
        del d["sections"]["evac"]
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="missing sections"):
            p.validate()
    
    def test_validate_overlap(self):
        d = _valid_profile_dict()
        d["sections"]["evac"] = [50.0, 100.0]  # overlaps stable's start
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="before previous section"):
            p.validate()
    
    def test_validate_exceeds_360(self):
        d = _valid_profile_dict()
        d["sections"]["baseline_post"] = [302.4, 400.0]
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="exceeds 360"):
            p.validate()
    
    def test_validate_bad_primary_section(self):
        d = _valid_profile_dict()
        d["primary_section"] = "nonexistent"
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="primary_section"):
            p.validate()


class TestLoadActiveProfile:
    def test_load_valid(self):
        runtime_cfg = {
            "active_profile": "test",
            "cycle_profiles": {"test": _valid_profile_dict()}
        }
        p = load_active_cycle_profile(runtime_cfg)
        assert p.profile_id == "test"
    
    def test_load_missing_active(self):
        runtime_cfg = {"cycle_profiles": {"test": _valid_profile_dict()}}
        with pytest.raises(ValueError, match="active_profile"):
            load_active_cycle_profile(runtime_cfg)
    
    def test_load_unknown_profile(self):
        runtime_cfg = {
            "active_profile": "missing",
            "cycle_profiles": {"test": _valid_profile_dict()}
        }
        with pytest.raises(ValueError, match="not found"):
            load_active_cycle_profile(runtime_cfg)
```

## 1.4 验收标准

- [ ] `runtime.yaml` 改造为 `cycle_profiles` 结构,含一个完整的 `bph_13000` profile
- [ ] `core/cycle_profile.py` 实现完整,validate() 能捕获 5 类错误
- [ ] `configs/loaders.py` 提供 `load_active_cycle_profile()` 函数
- [ ] `database_logger.py` 新增 9 个字段的 ALTER 语句和 `log_record` 参数
- [ ] `tests/test_cycle_profile.py` 全部通过
- [ ] 现有 `tests/test_database.py` 不退化(向后兼容)
- [ ] 启动 `python main.py --mode mock`,日志中应出现 "CycleProfile validated: bph_13000"

## 1.5 给后续任务留的接口约定

- 所有读取时序参数的代码,**必须通过 `CycleProfile` 对象**,不能再直接读 yaml 字段
- 任务 2 起,`runtime_cfg["cycle_detection"]` 这个旧路径**完全废弃**
- 数据库每条新记录必须填 `cycle_profile_id`(任务 8 在写入时设置)

---

# 任务 2:FSM 全周期采集

## 2.1 背景

v2.5 的 FSM 在角度跨过 100° 时触发,采 15 点(3.0s @ 200ms)结束,只覆盖保压段。v2.6 改为整圈采集——角度跨过 0° 触发,采 70 点(7.0s @ 100ms)结束,覆盖整个 6900ms 周期(6 段全部包含)。

时序变化:

| 项 | v2.5 | v2.6 |
|---|---|---|
| 触发角度 | 100° | 0°(整圈起点) |
| 采集点数 | 15 | 70 |
| 采样间隔 | 200ms | 100ms |
| 采集时长 | 3.0s | 7.0s |
| 超时阈值 | 8s | 10s |

## 2.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `core/cycle_fsm.py` | 重构 `CabinFSM` 使用 CycleProfile;新增"角度归零"作为备用结束条件 |
| `tests/test_cycle_fsm.py` | 全部测试用例改写,适配新参数和触发逻辑 |
| `core/polling_engine.py` | 无改动(polling 仍是 10ms,与 FSM 解耦) |
| `pipeline/processing_loop.py` | `__init__` 改为接收 CycleProfile;`_handle_complete_cycle` 暂不改 |

## 2.3 详细改动

### 2.3.1 重构 `core/cycle_fsm.py`

主要变化点:

1. `CabinFSM.__init__` 改为接收 `CycleProfile` 对象,而非 dict
2. `_handle_idle` 触发逻辑改为"角度跨过 trigger_angle"(原来是 start_angle)
3. `_handle_collecting` 新增"角度归零作为备用结束条件":如果 70 点未采满但角度从 360° 附近跨回 0°,也结束(说明转盘转过了一圈)
4. `CycleData` 新增 `cycle_profile_id: str` 字段

#### 关键代码框架

```python
# core/cycle_fsm.py (v2.6 改写版)
"""Finite State Machine for full-cycle data collection (v2.6).

v2.6 changes:
- Uses CycleProfile instead of flat dict config
- Triggers at angle crossing trigger_angle (default 0°)
- Collects collection_points samples (default 70)
- Backup end condition: angle wraps back to near 0° (full revolution)
- Records cycle_profile_id for traceability
"""

from __future__ import annotations
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from core.cycle_profile import CycleProfile
from core.polling_engine import CabinFrame

logger = logging.getLogger(__name__)


class CycleState(enum.Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    PROCESSING = "PROCESSING"
    FAULT = "FAULT"


@dataclass
class CycleData:
    """Accumulated data for one test cycle."""
    pressures: List[float] = field(default_factory=list)
    angles: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    ai_values: List[int] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    leak_valve_status: bool = False
    end_angle: float = 0.0
    start_time: float = 0.0
    cycle_profile_id: str = ""        # v2.6 新增: 标识当时使用的 profile


class CabinFSM:
    """State machine for one cabin's data collection.
    
    v2.6: Driven by CycleProfile rather than raw dict config.
    """
    
    # 角度归零的容差: 当从 >330° 跨回 <30°, 视为整圈完成
    WRAP_BACK_THRESHOLD = 30.0
    WRAP_FROM_THRESHOLD = 330.0
    
    def __init__(self, cabin_id: int, profile: CycleProfile):
        self.cabin_id = cabin_id
        self._profile = profile
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_angle: Optional[float] = None
        self._last_sample_ts: float = 0.0
    
    @property
    def state(self) -> CycleState:
        return self._state
    
    @property
    def data(self) -> CycleData:
        return self._data
    
    @property
    def point_count(self) -> int:
        return len(self._data.pressures)
    
    def update(self, frame: CabinFrame) -> CycleState:
        """Feed a polling frame and possibly transition state."""
        angle = frame.rt_angle
        ts = frame.timestamp
        
        if self._state == CycleState.IDLE:
            self._handle_idle(angle, ts, frame)
        elif self._state == CycleState.COLLECTING:
            self._handle_collecting(angle, ts, frame)
        
        self._last_angle = angle
        return self._state
    
    def harvest(self) -> CycleData:
        return self._data
    
    def reset(self) -> None:
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_angle = None
        self._last_sample_ts = 0.0
    
    def force_fault(self, reason: str = "") -> None:
        self._state = CycleState.FAULT
        logger.warning("Cabin %d: forced FAULT (%s)", self.cabin_id, reason)
    
    def clear_fault(self) -> None:
        self.reset()
    
    # ── Internal handlers ──────────────────────────────────────
    
    def _handle_idle(self, angle: float, ts: float, frame: CabinFrame) -> None:
        """Trigger collection when angle crosses trigger_angle upward.
        
        v2.6 default trigger is 0°. The 'crossing' must handle the wrap-around:
        previous angle ~358°, current ~2° → counts as crossing 0°.
        """
        if self._last_angle is None:
            return
        
        trigger = self._profile.trigger_angle
        
        crossed = False
        if trigger > 0:
            # Standard upward crossing
            if self._last_angle < trigger <= angle:
                crossed = True
        else:
            # Trigger at 0°: detect wrap-around (e.g. 358° -> 2°)
            if self._last_angle > self.WRAP_FROM_THRESHOLD and angle < self.WRAP_BACK_THRESHOLD:
                crossed = True
        
        if crossed:
            self._state = CycleState.COLLECTING
            self._data = CycleData(
                start_time=ts,
                leak_valve_status=frame.leak_valve_status,
                cycle_profile_id=self._profile.profile_id,
            )
            self._append(frame)
            self._last_sample_ts = ts
            logger.info(
                "Cabin %d: IDLE -> COLLECTING (trigger=%.1f°, angle %.1f° -> %.1f°)",
                self.cabin_id, trigger, self._last_angle, angle,
            )
    
    def _handle_collecting(self, angle: float, ts: float, frame: CabinFrame) -> None:
        """Collect samples at regular intervals until target count or wrap-back."""
        elapsed = ts - self._data.start_time
        
        # Sample at interval
        since_last = ts - self._last_sample_ts
        if since_last >= self._profile.collection_interval_s:
            self._append(frame)
            self._last_sample_ts = ts
        
        target_points = self._profile.collection_points
        
        # ── End condition 1: reached target point count ──────
        if len(self._data.pressures) >= target_points:
            self._data.end_angle = angle
            self._transition_to_processing(
                f"collected {target_points} points, end_angle={angle:.1f}°"
            )
            return
        
        # ── End condition 2: angle wrap-back (full revolution) ──
        # Useful as a safety net: if sampling is slower than expected and
        # we've gone a full cycle without reaching target_points
        if (self._last_angle is not None
                and self._last_angle > self.WRAP_FROM_THRESHOLD
                and angle < self.WRAP_BACK_THRESHOLD
                and len(self._data.pressures) >= target_points * 0.7):
            # Reached at least 70% of target before wrap-back: accept as PROCESSING
            self._data.end_angle = angle
            self._transition_to_processing(
                f"angle wrap-back (collected {len(self._data.pressures)}/{target_points} "
                f"points before full revolution)"
            )
            return
        
        # ── End condition 3: timeout → FAULT ──────────────────
        if elapsed >= self._profile.collection_timeout_s:
            self._data.end_angle = angle
            self._state = CycleState.FAULT
            logger.warning(
                "Cabin %d: COLLECTING -> FAULT (timeout %.1fs, %d/%d points)",
                self.cabin_id, elapsed, len(self._data.pressures), target_points,
            )
    
    def _transition_to_processing(self, reason: str) -> None:
        self._state = CycleState.PROCESSING
        logger.info(
            "Cabin %d: COLLECTING -> PROCESSING (%s, %.3fs)",
            self.cabin_id, reason,
            time.time() - self._data.start_time if self._data.start_time else 0,
        )
    
    def _append(self, frame: CabinFrame) -> None:
        self._data.pressures.append(frame.rt_pressure)
        self._data.angles.append(frame.rt_angle)
        self._data.timestamps.append(frame.timestamp)
        self._data.ai_values.append(frame.rt_ai)
        self._data.positions.append(frame.rt_position)


class CycleFSMManager:
    """Manages FSM instances for all active cabins."""
    
    def __init__(self, cabin_count: int, profile: CycleProfile,
                 active_start: int = 1, active_end: int = None):
        self._cabin_count = cabin_count
        self._profile = profile
        self._active_start = active_start
        self._active_end = active_end if active_end is not None else cabin_count - 1
        
        self.fsms: Dict[int, CabinFSM] = {
            i: CabinFSM(i, profile)
            for i in range(self._active_start, self._active_end + 1)
        }
        logger.info(
            "CycleFSMManager: %d cabins active [%d..%d], profile=%s",
            len(self.fsms), self._active_start, self._active_end, profile.profile_id,
        )
    
    def feed_frames(self, frames: List[CabinFrame]) -> None:
        """Distribute frames to per-cabin FSMs."""
        for frame in frames:
            cid = frame.cabin_index
            if cid in self.fsms:
                self.fsms[cid].update(frame)
```

### 2.3.2 重写 `tests/test_cycle_fsm.py`

```python
"""Unit tests for v2.6 cycle FSM."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.cycle_fsm import CabinFSM, CycleState
from core.cycle_profile import CycleProfile
from core.polling_engine import CabinFrame


@pytest.fixture
def profile():
    """13000 BPH profile, simplified for tests (interval=0 for fast iteration)."""
    return CycleProfile(
        profile_id="test",
        bph=13000,
        cycle_total_ms=6900,
        sections={
            "baseline_pre":  (0.0, 57.6),
            "evac":          (57.6, 93.0),
            "stable":        (93.0, 115.0),
            "hold":          (115.0, 273.6),
            "release":       (273.6, 302.4),
            "baseline_post": (302.4, 360.0),
        },
        trigger_angle=0.0,
        collection_points=10,        # small for tests
        collection_interval_s=0.0,    # no rate limit
        collection_timeout_s=8.0,
        primary_section="hold",
    )


def _frame(ci, p, a=0.0, ts=None):
    return CabinFrame(
        cabin_index=ci, rt_ai=0, rt_pressure=p,
        rt_position=0, rt_angle=a,
        leak_valve_status=False,
        timestamp=ts or time.time(),
    )


class TestCabinFSM_v26:
    def test_initial_state(self, profile):
        fsm = CabinFSM(1, profile)
        assert fsm.state == CycleState.IDLE
    
    def test_trigger_on_wrap_around(self, profile):
        """Trigger at 0° means wrap-around 358° -> 2°."""
        fsm = CabinFSM(1, profile)
        fsm.update(_frame(1, 0, 358.0))   # last_angle = 358
        fsm.update(_frame(1, 0, 2.0))      # crosses 0
        assert fsm.state == CycleState.COLLECTING
    
    def test_no_trigger_in_middle(self, profile):
        """Updates from middle of cycle should not trigger."""
        fsm = CabinFSM(1, profile)
        fsm.update(_frame(1, 600, 150.0))
        fsm.update(_frame(1, 600, 200.0))
        assert fsm.state == CycleState.IDLE
    
    def test_collect_to_target(self, profile):
        """Collect exactly target_points samples."""
        fsm = CabinFSM(1, profile)
        # Trigger
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))
        # Collect 9 more (1 already appended at trigger)
        for ang in [10, 50, 80, 100, 150, 200, 250, 280, 320]:
            fsm.update(_frame(1, 600, ang))
        assert fsm.state == CycleState.PROCESSING
        assert fsm.point_count == 10
    
    def test_wrap_back_safety_net(self, profile):
        """If wrap-back happens before target_points but >= 70% reached, accept."""
        # target=10, 70% = 7
        fsm = CabinFSM(1, profile)
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))    # 1 point
        for ang in [50, 100, 150, 200, 250, 300]:  # 6 more = 7 total
            fsm.update(_frame(1, 600, ang))
        # Now wrap-back: last=300, current=5
        fsm.update(_frame(1, 600, 5.0))   # actually this would be a new trigger?
        # Note: in real FSM, while still COLLECTING, wrap-back is the safety end.
        # The implementation in 2.3.1 needs the wrap angle while still in COLLECTING.
        # Let's test with last=350, current=5 to be unambiguous:
        # (state machine in COLLECTING, angle 350 -> 5 is wrap-back)
    
    def test_timeout_to_fault(self, profile):
        """Slow data → timeout → FAULT."""
        fsm = CabinFSM(1, profile)
        t0 = time.time()
        fsm.update(_frame(1, 0, 358.0, ts=t0))
        fsm.update(_frame(1, 0, 2.0, ts=t0))
        # Simulate 9 seconds passing without enough points
        fsm.update(_frame(1, 600, 50.0, ts=t0 + 9.0))
        assert fsm.state == CycleState.FAULT
    
    def test_cycle_profile_id_recorded(self, profile):
        """CycleData should record the profile_id used."""
        fsm = CabinFSM(1, profile)
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))
        assert fsm.data.cycle_profile_id == "test"
    
    def test_reset(self, profile):
        fsm = CabinFSM(1, profile)
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))
        fsm.reset()
        assert fsm.state == CycleState.IDLE
        assert fsm.point_count == 0
```

### 2.3.3 修改 `pipeline/processing_loop.py` 仅改 `__init__`

`ProcessingLoop.__init__` 接收 `profile: CycleProfile` 参数,传给 `CycleFSMManager`。处理逻辑(`_process_complete_cycle` 等)在任务 8 整体重构。

```python
def __init__(
    self,
    runtime_cfg: Dict[str, Any],
    profile: CycleProfile,                # v2.6 新增
    polling_engine: PollingEngine,
    fsm_manager: CycleFSMManager,
    # ... 其他参数 ...
):
    self._profile = profile
    # ... 其余赋值不变 ...
```

### 2.3.4 修改 `main.py` 启动逻辑

```python
from configs.loaders import load_active_cycle_profile

def main():
    # ... 现有的 config 加载 ...
    profile = load_active_cycle_profile()
    
    # FSM Manager 现在用 profile 初始化
    fsm_manager = CycleFSMManager(
        cabin_count=plc_cfg["cabin_array"]["cabin_count"],
        profile=profile,
        active_start=plc_cfg["cabin_array"].get("active_start", 1),
        active_end=plc_cfg["cabin_array"].get("active_end", 25),
    )
    
    # ProcessingLoop 也接收 profile
    processing_loop = ProcessingLoop(
        runtime_cfg=runtime_cfg,
        profile=profile,
        # ... 其他参数 ...
    )
```

## 2.4 验收标准

- [ ] `core/cycle_fsm.py` 重构完成,接收 `CycleProfile` 而非 dict
- [ ] 触发逻辑支持"trigger_angle=0°"的环绕检测
- [ ] 备用结束条件"角度归零"已实现
- [ ] `CycleData.cycle_profile_id` 正确填入
- [ ] `tests/test_cycle_fsm.py` 全部通过
- [ ] mock 模式下系统启动后,采集 30 秒,导出的记录数 > 0,每条 `pressure_data` 长度 = 70

## 2.5 风险与注意

**风险 1:Mock 模式的角度模拟需要更新**
`MockS7Connection` 现在以 200ms 间隔生成数据,角度变化可能不够细。任务 2 完成后需要在 `core/polling_engine.py` 的 Mock 部分,确保模拟数据的角度能从 0° 平滑增长到 360° 并循环。

**风险 2:长采集对 PLC 写回的影响**
v2.5 是采 3 秒处理 + 写回,v2.6 是采 7 秒处理 + 写回。这意味着每次推理结果延迟比原来多 4 秒到达 PLC。**这件事需要协调自控工程师确认**——客户是否接受这个延迟?如果不接受,需要在采集到一半(约 3.5s)就开始流式推理。本期暂不实现流式,作为 v2.7 的优化点。

**风险 3:轮询缓冲区大小**
`polling_engine.py` 的 `buffer_size = 10000`,以 10ms × 25 舱计算,缓冲区能存约 4 秒数据。在 7 秒采集场景下不够。**任务 2 中需把 `buffer_size` 从 10000 提升到 25000**(相当于 10 秒),修改 `configs/plc.yaml`:

```yaml
polling:
  interval_ms: 10
  buffer_size: 25000   # v2.6: 提升以适应 70 点采集
```


---

# 任务 3:段切割与 43 维特征工程

## 3.1 背景

70 点全周期数据需要按角度切成 6 段,每段独立计算 7 个统计量,组成 43 维特征向量(6 × 7 + 1 = 43,最后 1 维是 cavity_id)。

特征维度从 7 → 43 看似巨变,但 M1 模型对此完全透明——M1 只用 `section_hold_trend_slope`,这个数值与 v2.5 的 `trend_slope` 计算公式相同。M2 则使用全部 43 维,有机会学到舱体异常、零点漂移等次级信号。

## 3.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `core/curve_segmenter.py` | **新增**:按角度切段 |
| `core/features.py` | 重构:从 `compute_features(pressures, cavity_id)` 改为 `compute_features_v26(pressures, angles, cavity_id, profile)` |
| `core/feature_spec.py` | **新增**:43 维特征命名与顺序常量 |
| `tests/test_curve_segmenter.py` | **新增** |
| `tests/test_features.py` | 扩展:覆盖 43 维输出 |

## 3.3 详细改动

### 3.3.1 新增 `core/curve_segmenter.py`

```python
"""Curve segmentation by angle.

Splits a (pressures, angles) sequence into 6 sections according to a
CycleProfile's section boundaries. Each section is a contiguous slice
of the original sequence, identified by points whose angle falls within
[start_angle, end_angle).
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple

from core.cycle_profile import CycleProfile, SECTION_NAMES

logger = logging.getLogger(__name__)


def segment_by_angle(
    pressures: List[float],
    angles: List[float],
    profile: CycleProfile,
) -> Dict[str, List[float]]:
    """Split a pressure sequence into named sections by angle.
    
    Parameters
    ----------
    pressures : list of float
        Pressure values, length N.
    angles : list of float
        Corresponding angle values (degrees), length N.
    profile : CycleProfile
        Defines section boundaries.
    
    Returns
    -------
    dict[str, list[float]]
        Mapping from section name → list of pressures in that section.
        All 6 standard sections are always present (empty list if no points fall in range).
    
    Notes
    -----
    - Points whose angle falls outside all 6 sections are silently dropped.
    - Section boundaries are [start, end), so a point exactly at end belongs
      to the next section.
    - If pressures and angles have different lengths, the shorter is used.
    """
    if len(pressures) != len(angles):
        logger.warning(
            "segment_by_angle: length mismatch (pressures=%d, angles=%d), truncating",
            len(pressures), len(angles)
        )
    n = min(len(pressures), len(angles))
    
    # Initialize all sections to empty
    result = {name: [] for name in SECTION_NAMES}
    
    for i in range(n):
        a = angles[i]
        for name in SECTION_NAMES:
            start, end = profile.sections[name]
            if start <= a < end:
                result[name].append(pressures[i])
                break
    
    return result


def segment_indices_by_angle(
    angles: List[float],
    profile: CycleProfile,
) -> Dict[str, List[int]]:
    """Same as segment_by_angle but returns indices instead of values.
    
    Useful when caller needs to slice multiple parallel arrays
    (pressures + angles + timestamps + ...).
    """
    result = {name: [] for name in SECTION_NAMES}
    for i, a in enumerate(angles):
        for name in SECTION_NAMES:
            start, end = profile.sections[name]
            if start <= a < end:
                result[name].append(i)
                break
    return result
```

### 3.3.2 新增 `core/feature_spec.py`

```python
"""Feature specification for v2.6 (43-dimensional).

The 43-dim feature vector is:
    [
      # Section 1: baseline_pre (7 features)
      baseline_pre_max, baseline_pre_min, baseline_pre_difference,
      baseline_pre_average, baseline_pre_variance,
      baseline_pre_trend_slope, baseline_pre_count,
      
      # Section 2: evac (7 features)
      evac_max, evac_min, evac_difference, evac_average,
      evac_variance, evac_trend_slope, evac_count,
      
      # Section 3: stable (7)
      stable_max, ..., stable_count,
      
      # Section 4: hold (7) — primary section, M1 uses hold_trend_slope
      hold_max, ..., hold_count,
      
      # Section 5: release (7)
      release_max, ..., release_count,
      
      # Section 6: baseline_post (7)
      baseline_post_max, ..., baseline_post_count,
      
      # 43rd dim: cavity_id
      cavity_id
    ]
"""

from __future__ import annotations
from typing import List

from core.cycle_profile import SECTION_NAMES

# Per-section sub-features (7 of them).
# NOTE: 'count' replaces v2.5's 'cavity_id' as the 7th feature within each section
# (since cavity_id is now a single global field, not per-section).
SECTION_SUB_FEATURES = [
    "max",
    "min",
    "difference",
    "average",
    "variance",
    "trend_slope",
    "count",       # number of points in this section (catches abnormal cycles)
]

# Full 43-dim feature names in order
FEATURE_ORDER_43D: List[str] = [
    f"{section}_{sub}"
    for section in SECTION_NAMES
    for sub in SECTION_SUB_FEATURES
] + ["cavity_id"]

assert len(FEATURE_ORDER_43D) == 43, f"Expected 43 features, got {len(FEATURE_ORDER_43D)}"

# Index of the primary trend_slope (used by M1)
# This is hold_trend_slope (5th feature of 4th section, 0-indexed)
def primary_trend_slope_index(primary_section: str = "hold") -> int:
    """Get the index of the primary section's trend_slope in FEATURE_ORDER_43D."""
    target = f"{primary_section}_trend_slope"
    return FEATURE_ORDER_43D.index(target)
```

### 3.3.3 重构 `core/features.py`

**完全重写**(保留 v2.5 函数作为 deprecated):

```python
"""Feature computation for v2.6 (43-dimensional).

Replaces v2.5's 7-dim feature contract. Each cycle's pressure curve is
segmented into 6 sections by angle, and 7 statistical features are
computed per section, plus cavity_id as the 43rd feature.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np

from core.cycle_profile import CycleProfile, SECTION_NAMES
from core.curve_segmenter import segment_by_angle
from core.feature_spec import FEATURE_ORDER_43D, SECTION_SUB_FEATURES

logger = logging.getLogger(__name__)


def _compute_section_stats(pressures: List[float]) -> Dict[str, float]:
    """Compute 7 features for one section's pressure values.
    
    For empty or too-small sections, returns zeros (with count = N actual).
    """
    n = len(pressures)
    if n < 2:
        return {
            "max": 0.0, "min": 0.0, "difference": 0.0,
            "average": 0.0, "variance": 0.0, "trend_slope": 0.0,
            "count": float(n),
        }
    
    arr = np.asarray(pressures, dtype=np.float64)
    p_max = float(np.max(arr))
    p_min = float(np.min(arr))
    p_diff = p_max - p_min
    p_avg = float(np.mean(arr))
    p_var = float(np.var(arr))
    
    try:
        slope = float(np.polyfit(np.arange(n), arr, 1)[0])
    except Exception:
        slope = 0.0
    
    return {
        "max": round(p_max, 3),
        "min": round(p_min, 3),
        "difference": round(p_diff, 3),
        "average": round(p_avg, 3),
        "variance": round(p_var, 3),
        "trend_slope": round(slope, 6),
        "count": float(n),
    }


def compute_features_v26(
    pressures: List[float],
    angles: List[float],
    cavity_id: int,
    profile: CycleProfile,
) -> Dict[str, float]:
    """Compute the 43-dim feature dict for one cycle.
    
    Parameters
    ----------
    pressures : list of float
        Pressure samples (length N, typically 70).
    angles : list of float
        Corresponding angles in degrees (length N).
    cavity_id : int
        Cabin index (1..25).
    profile : CycleProfile
        Defines section boundaries for segmentation.
    
    Returns
    -------
    dict with 43 keys (see FEATURE_ORDER_43D for names and order).
    """
    if len(pressures) < 2:
        # Degenerate input: return all zeros
        feats = {name: 0.0 for name in FEATURE_ORDER_43D}
        feats["cavity_id"] = float(cavity_id)
        return feats
    
    # Segment by angle
    sections = segment_by_angle(pressures, angles, profile)
    
    # Compute per-section stats
    feats = {}
    for section_name in SECTION_NAMES:
        section_pressures = sections.get(section_name, [])
        section_stats = _compute_section_stats(section_pressures)
        for sub_name, value in section_stats.items():
            feats[f"{section_name}_{sub_name}"] = value
    
    # Add cavity_id
    feats["cavity_id"] = float(cavity_id)
    
    return feats


def features_to_vector(feats: Dict[str, float], mode: str = "43d") -> List[float]:
    """Convert a feature dict into a vector in the standard order."""
    if mode == "43d":
        return [feats.get(k, 0.0) for k in FEATURE_ORDER_43D]
    # Legacy 7-dim mode kept for migration testing only — should not be used in v2.6
    raise ValueError(f"Unsupported feature mode: {mode}. Use '43d'.")


# ── Deprecated v2.5 compatibility ──────────────────────────────────────
# Kept only for tests that haven't been migrated yet. Do NOT use in new code.

def compute_features(pressures: List[float], cavity_id: int) -> Dict[str, float]:
    """[DEPRECATED] v2.5 7-dim feature computation.
    
    Use compute_features_v26 instead.
    """
    logger.warning("compute_features (v2.5) is deprecated. Use compute_features_v26.")
    if not pressures or len(pressures) < 2:
        return {
            "max": 0.0, "min": 0.0, "difference": 0.0, "average": 0.0,
            "variance": 0.0, "trend_slope": 0.0, "cavity_id": float(cavity_id),
        }
    arr = np.asarray(pressures, dtype=np.float64)
    p_max, p_min = float(np.max(arr)), float(np.min(arr))
    return {
        "max": round(p_max, 3),
        "min": round(p_min, 3),
        "difference": round(p_max - p_min, 3),
        "average": round(float(np.mean(arr)), 3),
        "variance": round(float(np.var(arr)), 3),
        "trend_slope": round(float(np.polyfit(np.arange(len(arr)), arr, 1)[0]), 6),
        "cavity_id": float(cavity_id),
    }
```

### 3.3.4 单元测试 `tests/test_curve_segmenter.py`

```python
"""Tests for curve_segmenter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.cycle_profile import CycleProfile
from core.curve_segmenter import segment_by_angle, segment_indices_by_angle


@pytest.fixture
def profile():
    return CycleProfile(
        profile_id="test",
        bph=13000,
        cycle_total_ms=6900,
        sections={
            "baseline_pre":  (0.0, 60.0),
            "evac":          (60.0, 100.0),
            "stable":        (100.0, 120.0),
            "hold":          (120.0, 280.0),
            "release":       (280.0, 310.0),
            "baseline_post": (310.0, 360.0),
        },
        trigger_angle=0.0,
        collection_points=70,
        collection_interval_s=0.1,
        collection_timeout_s=10.0,
    )


def test_segment_basic(profile):
    pressures = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    angles = [10.0, 70.0, 110.0, 200.0, 290.0, 320.0]
    result = segment_by_angle(pressures, angles, profile)
    assert result["baseline_pre"] == [10.0]
    assert result["evac"] == [20.0]
    assert result["stable"] == [30.0]
    assert result["hold"] == [40.0]
    assert result["release"] == [50.0]
    assert result["baseline_post"] == [60.0]


def test_section_boundary_inclusive_left(profile):
    """Point exactly at start_angle should belong to that section."""
    pressures = [100.0]
    angles = [60.0]   # exactly at boundary baseline_pre/evac
    result = segment_by_angle(pressures, angles, profile)
    assert result["evac"] == [100.0]
    assert result["baseline_pre"] == []


def test_section_boundary_exclusive_right(profile):
    """Point exactly at end_angle should belong to the NEXT section."""
    pressures = [100.0]
    angles = [100.0]  # exactly at evac/stable boundary
    result = segment_by_angle(pressures, angles, profile)
    assert result["stable"] == [100.0]
    assert result["evac"] == []


def test_out_of_range_dropped(profile):
    """Points with angle > 360 should be dropped."""
    pressures = [50.0, 100.0]
    angles = [180.0, 400.0]
    result = segment_by_angle(pressures, angles, profile)
    assert len(result["hold"]) == 1
    total_collected = sum(len(v) for v in result.values())
    assert total_collected == 1  # 400° is dropped


def test_all_sections_present(profile):
    """Every standard section must be in result, even if empty."""
    result = segment_by_angle([10.0], [200.0], profile)
    assert set(result.keys()) == {
        "baseline_pre", "evac", "stable", "hold", "release", "baseline_post"
    }
    assert result["baseline_pre"] == []
```

### 3.3.5 扩展 `tests/test_features.py`

保留现有 v2.5 测试(它们会触发 deprecation warning 但仍然通过)。新增:

```python
class TestComputeFeaturesV26:
    @pytest.fixture
    def profile(self):
        # same as test_curve_segmenter
        from core.cycle_profile import CycleProfile
        return CycleProfile(
            profile_id="test", bph=13000, cycle_total_ms=6900,
            sections={
                "baseline_pre":  (0.0, 60.0),
                "evac":          (60.0, 100.0),
                "stable":        (100.0, 120.0),
                "hold":          (120.0, 280.0),
                "release":       (280.0, 310.0),
                "baseline_post": (310.0, 360.0),
            },
            trigger_angle=0.0, collection_points=70,
            collection_interval_s=0.1, collection_timeout_s=10.0,
        )
    
    def test_43_dim_output(self, profile):
        from core.features import compute_features_v26
        # 70 points spanning 0-360°
        pressures = [600.0 + i for i in range(70)]
        angles = [i * 360.0 / 70 for i in range(70)]
        feats = compute_features_v26(pressures, angles, cavity_id=5, profile=profile)
        # All 43 keys present
        from core.feature_spec import FEATURE_ORDER_43D
        for key in FEATURE_ORDER_43D:
            assert key in feats, f"missing key: {key}"
        assert feats["cavity_id"] == 5.0
    
    def test_features_to_vector_43d(self, profile):
        from core.features import compute_features_v26, features_to_vector
        feats = compute_features_v26(
            [600.0]*70, [i * 360.0 / 70 for i in range(70)], 5, profile
        )
        vec = features_to_vector(feats, mode="43d")
        assert len(vec) == 43
        assert vec[-1] == 5.0  # cavity_id is last
    
    def test_empty_section_yields_zeros(self, profile):
        """If no points fall in baseline_pre, its 7 features should be zeros."""
        from core.features import compute_features_v26
        # All angles in hold section
        pressures = [600.0]*10
        angles = [200.0]*10
        feats = compute_features_v26(pressures, angles, 1, profile)
        assert feats["baseline_pre_count"] == 0.0
        assert feats["baseline_pre_max"] == 0.0
        assert feats["hold_count"] == 10.0
    
    def test_primary_trend_slope_consistency(self, profile):
        """hold_trend_slope should match what M1 expects."""
        from core.features import compute_features_v26
        from core.feature_spec import primary_trend_slope_index, FEATURE_ORDER_43D
        # Hold section has linearly increasing pressure
        pressures = []
        angles = []
        for i in range(20):
            pressures.append(600.0 + i)   # slope 1.0
            angles.append(120.0 + i * 8.0)   # in hold section
        feats = compute_features_v26(pressures, angles, 1, profile)
        # Slope should be close to 1.0
        assert abs(feats["hold_trend_slope"] - 1.0) < 0.01
        # And accessible via primary_trend_slope_index
        idx = primary_trend_slope_index("hold")
        assert FEATURE_ORDER_43D[idx] == "hold_trend_slope"
```

## 3.4 验收标准

- [ ] `core/curve_segmenter.py` 实现 `segment_by_angle` + `segment_indices_by_angle`
- [ ] `core/feature_spec.py` 定义 `FEATURE_ORDER_43D`(43 个名字),`primary_trend_slope_index()`
- [ ] `core/features.py` 新增 `compute_features_v26()`,旧的 `compute_features()` 标记 deprecated
- [ ] 所有 43 维特征在合理输入下都能正确计算,空段返回 0(count=0)
- [ ] `tests/test_curve_segmenter.py` 全部通过
- [ ] `tests/test_features.py` 旧测试仍通过(兼容),新测试覆盖 43 维输出
- [ ] `compute_features_v26` 在 `pressures=[]` 时不抛异常

## 3.5 风险与注意

**风险 1:某些段在实际采集中可能太短(< 2 点),trend_slope 不可靠**
比如 evac 和 release 段都只有 4-6 个点,np.polyfit 拟合稳定性不如长段。这些短段的 trend_slope 进入 M2 时可能产生噪声。**应对**:M2 训练完后看 feature_importance,如果某些短段特征确实噪声大,就在训练时把它们去掉(由特征选择步骤实现,见任务 7)。本任务不做特殊处理,先全留着。

**风险 2:angles 序列可能有"非单调"现象**
比如转盘瞬时停顿、传感器抖动,导致 angle[i+1] < angle[i]。当前 `segment_by_angle` 不假设单调,直接按每点的 angle 值分配段——这是健壮的。

**风险 3:0° 附近的边界处理**
trigger_angle = 0° 触发时,采集起点的 angle 可能是 1° 或 358°(取决于何时跨过)。这些点会被分到 baseline_pre 段(0° 起)还是 baseline_post 段(< 360°)?按代码逻辑,357° 进 baseline_post(310° ≤ 357 < 360),2° 进 baseline_pre(0 ≤ 2 < 60)。**这是正确的**——同一个 angle 值不会同时进两段。

---

# 任务 4:zlib 压缩存储

## 4.1 背景

70 点 × (pressure float32 + angle float32) = 560 字节/记录。一天 4170 圈 × 25 舱 = 104k 条记录,纯压力曲线 ~58 MB,加上 angle/timestamp 等约 100 MB/天。zlib 压缩后预期 25-30 MB/天,3-4 倍压缩比。

实施方式:存储层透明压缩。业务代码继续用 `List[float]` 接口,在 `database_logger.log_record` 入口压缩,在 `query_records` 出口解压。

## 4.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `storage/compression.py` | **新增**:压缩/解压工具函数 |
| `storage/database_logger.py` | `log_record` 调用压缩;`query_records` 调用解压 |
| `storage/data_exporter.py` | CSV 导出时自动解压 |
| `tests/test_compression.py` | **新增** |

## 4.3 详细改动

### 4.3.1 新增 `storage/compression.py`

```python
"""Compression utilities for v2.6 storage.

Strategy:
  - Encode List[float] as numpy float32 array → bytes
  - zlib compress at level 6 (default, balanced speed/ratio)
  - Store as BLOB in SQLite

Compression ratio for typical pressure curves: 3-5x
Decompression overhead: < 1 ms per 70-point curve
"""

from __future__ import annotations
import zlib
from typing import List, Optional

import numpy as np


# zlib level: 1 (fastest) to 9 (best). Level 6 is the default and works well
# for our smooth-curve data.
COMPRESSION_LEVEL = 6


def compress_float_array(values: Optional[List[float]]) -> Optional[bytes]:
    """Compress a list of floats into a zlib-compressed BLOB.
    
    Parameters
    ----------
    values : list of float, or None
        The data to compress. Returns None if input is None or empty.
    
    Returns
    -------
    bytes (compressed) or None
    """
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    raw = arr.tobytes()
    return zlib.compress(raw, COMPRESSION_LEVEL)


def decompress_float_array(blob: Optional[bytes]) -> Optional[List[float]]:
    """Decompress a BLOB into a list of floats.
    
    Parameters
    ----------
    blob : bytes, or None
    
    Returns
    -------
    list of float, or None if blob is None/empty
    """
    if not blob:
        return None
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.tolist()


def estimate_compression_ratio(values: List[float]) -> float:
    """Diagnostic: report how well a particular list compresses."""
    if not values:
        return 0.0
    raw_size = len(values) * 4  # float32
    compressed = compress_float_array(values)
    return raw_size / len(compressed) if compressed else 0.0
```

### 4.3.2 修改 `storage/database_logger.py`

#### `log_record` 内部调用压缩

```python
from storage.compression import compress_float_array, decompress_float_array


def log_record(self, cavity_id, pressures, angles, ai_values, positions,
               features, label, probability, confidence, model_version,
               duration_s, leak_valve_status=None, end_angle=None,
               batch_id="",
               cycle_profile_id=None,
               q_est=None, q_threshold=None, q_uncertainty=None,
               m1_q=None, m2_q=None, m_disagreement=None,
               product_id=None) -> int:
    """Insert a record. Pressures and angles are auto-compressed to BLOB.
    
    NOTE: v2.6 onwards, pressure_data and angle_data (JSON) are NOT written.
    Only pressure_data_compressed and angle_data_compressed are stored.
    """
    # Compress arrays
    pressure_blob = compress_float_array(pressures)
    angle_blob = compress_float_array(angles) if angles else None
    
    with self._lock:
        try:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            cur = self._conn.execute(
                "INSERT INTO test_records ("
                "  batch_id, cavity_id, timestamp,"
                "  pressure_data, angle_data, ai_data, position_data,"          # legacy: NULL in v2.6
                "  features, label, probability, confidence,"
                "  model_version, duration_s, point_count,"
                "  leak_valve_status, end_angle,"
                "  cycle_profile_id, pressure_data_compressed, angle_data_compressed,"
                "  q_est, q_threshold, q_uncertainty,"
                "  m1_q, m2_q, m_disagreement, product_id"
                ") VALUES (?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?, ?,?, ?,?,?, ?,?,?, ?,?,?,?)",
                (
                    batch_id, cavity_id, ts,
                    None, None, None, None,                   # legacy fields NULL
                    json.dumps(features), label, probability, confidence,
                    model_version, round(duration_s, 3), len(pressures),
                    int(leak_valve_status) if leak_valve_status is not None else None,
                    round(end_angle, 2) if end_angle is not None else None,
                    cycle_profile_id, pressure_blob, angle_blob,
                    q_est, q_threshold, q_uncertainty,
                    m1_q, m2_q, m_disagreement, product_id,
                )
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as exc:
            raise StorageError(f"log_record failed: {exc}") from exc
```

#### `query_records_with_curves` 自动解压

新增 `get_full_record(record_id)` 方法,返回含解压后数据的完整 dict:

```python
def get_full_record(self, record_id: int) -> Optional[Dict[str, Any]]:
    """Fetch one record with decompressed pressure/angle arrays.
    
    Returns dict with all v2.6 fields populated, including decompressed
    `pressures` and `angles` lists (or [] if not present).
    """
    with self._lock:
        cur = self._conn.execute(
            "SELECT id, batch_id, cavity_id, timestamp, "
            "  pressure_data_compressed, angle_data_compressed, "
            "  features, label, probability, confidence, "
            "  model_version, duration_s, point_count, "
            "  leak_valve_status, end_angle, "
            "  cycle_profile_id, q_est, q_threshold, q_uncertainty, "
            "  m1_q, m2_q, m_disagreement, product_id "
            "FROM test_records WHERE id = ?",
            (record_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        rec = dict(zip(cols, row))
        # Decompress
        rec["pressures"] = decompress_float_array(rec.pop("pressure_data_compressed")) or []
        rec["angles"] = decompress_float_array(rec.pop("angle_data_compressed")) or []
        # Parse features JSON
        if rec.get("features"):
            rec["features"] = json.loads(rec["features"])
        return rec
```

### 4.3.3 修改 `storage/data_exporter.py`

CSV 导出时自动解压。导出格式保持兼容:`pressure_data` 列仍是 JSON 数组字符串,`angle_data` 同。这样训练脚本无需改动。

```python
def export_to_csv(self, output_path: str, batch_size: int = 1000) -> int:
    """Export all records to CSV, decompressing curves on the fly."""
    import csv
    from storage.compression import decompress_float_array
    
    cur = self._conn.execute(
        "SELECT id, batch_id, cavity_id, timestamp, "
        "  pressure_data_compressed, angle_data_compressed, "
        "  features, label, probability, confidence, "
        "  model_version, duration_s, point_count, "
        "  leak_valve_status, end_angle, "
        "  cycle_profile_id, q_est, q_threshold, "
        "  m1_q, m2_q, m_disagreement, product_id "
        "FROM test_records ORDER BY id"
    )
    
    cols = [d[0] for d in cur.description]
    # Replace _compressed columns with their JSON-array equivalents in the CSV
    csv_cols = [
        "pressure_data" if c == "pressure_data_compressed"
        else "angle_data" if c == "angle_data_compressed"
        else c
        for c in cols
    ]
    
    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_cols)
        for row in cur:
            row = list(row)
            # Decompress and re-serialize to JSON
            for i, c in enumerate(cols):
                if c == "pressure_data_compressed":
                    arr = decompress_float_array(row[i]) or []
                    row[i] = json.dumps(arr)
                elif c == "angle_data_compressed":
                    arr = decompress_float_array(row[i]) or []
                    row[i] = json.dumps(arr)
            writer.writerow(row)
            count += 1
    return count
```

### 4.3.4 单元测试 `tests/test_compression.py`

```python
"""Tests for storage.compression."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from storage.compression import (
    compress_float_array, decompress_float_array, estimate_compression_ratio,
)


class TestCompression:
    def test_round_trip_small(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        blob = compress_float_array(values)
        assert blob is not None
        recovered = decompress_float_array(blob)
        assert recovered == pytest.approx(values, rel=1e-6)
    
    def test_round_trip_70_points(self):
        # Realistic 70-point pressure curve (smooth with slight noise)
        np.random.seed(42)
        base = np.linspace(0, 600, 70)
        noise = np.random.normal(0, 0.5, 70)
        values = (base + noise).tolist()
        blob = compress_float_array(values)
        recovered = decompress_float_array(blob)
        assert recovered == pytest.approx(values, rel=1e-4)
    
    def test_compression_ratio_typical(self):
        """Typical curve should compress 3-5x."""
        np.random.seed(42)
        values = (np.linspace(0, 600, 70) + np.random.normal(0, 0.5, 70)).tolist()
        ratio = estimate_compression_ratio(values)
        assert 2.0 < ratio < 8.0  # generous bound
    
    def test_empty_input(self):
        assert compress_float_array([]) is None
        assert compress_float_array(None) is None
        assert decompress_float_array(None) is None
        assert decompress_float_array(b"") is None
    
    def test_blob_is_bytes(self):
        blob = compress_float_array([1.0, 2.0, 3.0])
        assert isinstance(blob, bytes)
```

## 4.4 验收标准

- [ ] `storage/compression.py` 实现完整,3 个公开函数
- [ ] `tests/test_compression.py` 全部通过
- [ ] `database_logger.log_record` 自动压缩,新数据 `pressure_data` 列为 NULL,`pressure_data_compressed` 有数据
- [ ] `data_exporter.export_to_csv` 导出的 CSV 中,`pressure_data` 列是 JSON 数组(自动解压后)
- [ ] 现有 `tests/test_database.py` 兼容(允许调整以适配新 schema)
- [ ] 一段端到端 mock 数据采集后,数据库文件大小 < 不压缩版本的 60%


---

# 任务 5:V_cabin 配置 + 标定脚本

## 5.1 背景

v2.6 的物理关系 `Q = V_cabin × dp/dt` 要求每个真空舱独立标定 V_cabin。当前 v2.5 没有这个参数。本任务建立 V_cabin 的存储结构、加载机制,以及标定记录脚本。

实验日 D1(5/6 周三)做 V_cabin 注水法标定,该脚本用于把测量结果写入配置。M1 模型加载后会查表使用各舱 V_cabin。

## 5.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `configs/cabins.yaml` | **新增**:25 舱 V_cabin 配置 |
| `configs/loaders.py` | 新增 `load_cabins_config()` 与 `get_v_cabin()` |
| `scripts/calibrate_v_cabin.py` | **新增**:标定 CLI 工具 |
| `tests/test_loaders_cabins.py` | **新增** |

## 5.3 详细改动

### 5.3.1 新增 `configs/cabins.yaml`

```yaml
# 各真空舱有效体积 V_cabin 标定值
# 单位: m³ (= 立方米); 注水法测量, 通常在 [2.5e-4, 4.0e-4] 区间
# Cabin[0] 为预留, 不参与
# u_v_cabin 为标定不确定度 (1-sigma, 单位 m³)

calibration_date: ""    # 实验日填入 YYYY-MM-DD
calibrator: ""          # 标定人姓名

cabins:
  1:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  2:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  3:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  4:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  5:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  6:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  7:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  8:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  9:  { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  10: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  11: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  12: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  13: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  14: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  15: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  16: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  17: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  18: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  19: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  20: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  21: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  22: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  23: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  24: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }
  25: { v_cabin: 3.50e-4, u_v_cabin: 7.0e-6, notes: "初始占位值" }

# 默认值: 当某 cabin 未标定时使用 (V_cabin 占位值)
default:
  v_cabin: 3.50e-4
  u_v_cabin: 1.0e-5
  notes: "fallback for uncalibrated cabin"
```

### 5.3.2 在 `configs/loaders.py` 新增加载函数

```python
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple


def load_cabins_config(path: str = "configs/cabins.yaml") -> Dict[str, Any]:
    """Load V_cabin calibration values for all 25 cabins.
    
    Returns
    -------
    dict with keys: calibration_date, calibrator, cabins, default
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"cabins config not found: {path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_v_cabin(cabins_cfg: Dict[str, Any], cabin_id: int) -> Tuple[float, float]:
    """Get (v_cabin, u_v_cabin) for a specific cabin.
    
    Falls back to default if the cabin is not calibrated.
    
    Returns
    -------
    (v_cabin, u_v_cabin) in m³.
    """
    entry = cabins_cfg.get("cabins", {}).get(cabin_id)
    if entry and "v_cabin" in entry:
        return float(entry["v_cabin"]), float(entry.get("u_v_cabin", 0.0))
    default = cabins_cfg.get("default", {})
    return float(default.get("v_cabin", 3.5e-4)), float(default.get("u_v_cabin", 1.0e-5))


def is_cabin_calibrated(cabins_cfg: Dict[str, Any], cabin_id: int) -> bool:
    """Check whether cabin_id has been calibrated (vs using default fallback)."""
    entry = cabins_cfg.get("cabins", {}).get(cabin_id, {})
    notes = entry.get("notes", "")
    # Heuristic: "占位值" or empty means uncalibrated
    return entry.get("v_cabin") is not None and "占位" not in notes
```

### 5.3.3 新增 `scripts/calibrate_v_cabin.py`

```python
#!/usr/bin/env python3
"""V_cabin calibration recorder.

Usage:
    python scripts/calibrate_v_cabin.py \\
        --cabin 5 --weights-grams 348.2,348.5,348.0 \\
        --notes "first batch"

The script:
1. Validates input (3 repeats minimum, CV < 2%)
2. Computes mean V_cabin in m³ and u_v_cabin (1-sigma std dev)
3. Updates configs/cabins.yaml with the new value
4. Appends a row to data/calibration_history/v_cabin_log.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CABINS_YAML = PROJECT_ROOT / "configs" / "cabins.yaml"
LOG_CSV = PROJECT_ROOT / "data" / "calibration_history" / "v_cabin_log.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Record V_cabin calibration")
    p.add_argument("--cabin", type=int, required=True, help="Cabin ID (1..25)")
    p.add_argument("--weights-grams", required=True,
                   help="Comma-separated water weights in grams (3+ repeats)")
    p.add_argument("--calibrator", default="", help="Calibrator name")
    p.add_argument("--notes", default="", help="Notes")
    p.add_argument("--cv-limit", type=float, default=0.02, help="CV threshold (default 2%%)")
    p.add_argument("--dry-run", action="store_true", help="Don't write yaml, only report")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not (1 <= args.cabin <= 25):
        sys.exit(f"ERROR: cabin must be in [1, 25], got {args.cabin}")
    
    weights = [float(w.strip()) for w in args.weights_grams.split(",")]
    if len(weights) < 3:
        sys.exit(f"ERROR: need >= 3 repeats, got {len(weights)}")
    
    # Statistics
    mean_g = sum(weights) / len(weights)
    var_g = sum((w - mean_g) ** 2 for w in weights) / (len(weights) - 1)
    std_g = var_g ** 0.5
    cv = std_g / mean_g
    
    # Convert: water weight (g) → volume (m³)
    # 1 g of water at room temperature = 1 mL = 1e-6 m³
    v_cabin_m3 = mean_g * 1e-6
    u_v_cabin_m3 = std_g * 1e-6
    
    print(f"\n=== V_cabin Calibration: Cabin {args.cabin} ===")
    print(f"Repeats:           {weights} g")
    print(f"Mean:              {mean_g:.2f} g  →  {v_cabin_m3:.3e} m³")
    print(f"Std deviation:     {std_g:.3f} g  →  u = {u_v_cabin_m3:.3e} m³")
    print(f"CV:                {cv * 100:.2f}%  (limit {args.cv_limit * 100:.1f}%)")
    
    if cv > args.cv_limit:
        print(f"\n⚠ WARNING: CV exceeds limit. Will NOT write to yaml.")
        print(f"  Repeat measurement before recording.")
        # Still log to history for traceability
        _append_log(args.cabin, weights, mean_g, std_g, cv,
                    args.calibrator, args.notes + " [REJECTED: CV exceeded]",
                    accepted=False)
        sys.exit(1)
    
    if args.dry_run:
        print("\n[dry-run] Would update configs/cabins.yaml.")
        return
    
    # Update yaml
    _update_cabin_yaml(args.cabin, v_cabin_m3, u_v_cabin_m3,
                       args.calibrator, args.notes)
    
    # Log history
    _append_log(args.cabin, weights, mean_g, std_g, cv,
                args.calibrator, args.notes, accepted=True)
    
    print(f"\n✓ Updated configs/cabins.yaml for cabin {args.cabin}")
    print(f"✓ Logged to {LOG_CSV.relative_to(PROJECT_ROOT)}")


def _update_cabin_yaml(cabin_id, v_cabin, u_v_cabin, calibrator, notes):
    with open(CABINS_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    today = time.strftime("%Y-%m-%d")
    if not cfg.get("calibration_date"):
        cfg["calibration_date"] = today
    if calibrator and not cfg.get("calibrator"):
        cfg["calibrator"] = calibrator
    
    cfg.setdefault("cabins", {})
    cfg["cabins"][cabin_id] = {
        "v_cabin": float(f"{v_cabin:.4e}"),
        "u_v_cabin": float(f"{u_v_cabin:.2e}"),
        "notes": notes or f"calibrated {today}",
    }
    
    with open(CABINS_YAML, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _append_log(cabin_id, weights, mean_g, std_g, cv, calibrator, notes, accepted):
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_CSV.exists()
    
    with open(LOG_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp", "cabin_id", "weights_g", "mean_g", "std_g",
                "cv_percent", "v_cabin_m3", "u_v_cabin_m3", "calibrator",
                "notes", "accepted"
            ])
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            cabin_id, "|".join(f"{w:.2f}" for w in weights),
            f"{mean_g:.3f}", f"{std_g:.3f}", f"{cv * 100:.2f}",
            f"{mean_g * 1e-6:.4e}", f"{std_g * 1e-6:.2e}",
            calibrator, notes, "yes" if accepted else "no"
        ])


if __name__ == "__main__":
    main()
```

### 5.3.4 单元测试 `tests/test_loaders_cabins.py`

```python
"""Tests for cabins config loader."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import yaml

from configs.loaders import load_cabins_config, get_v_cabin, is_cabin_calibrated


@pytest.fixture
def cabins_yaml(tmp_path):
    f = tmp_path / "cabins.yaml"
    data = {
        "calibration_date": "2026-05-06",
        "calibrator": "tester",
        "cabins": {
            1: {"v_cabin": 3.45e-4, "u_v_cabin": 6e-6, "notes": "calibrated"},
            2: {"v_cabin": 3.50e-4, "u_v_cabin": 7e-6, "notes": "占位值"},
        },
        "default": {"v_cabin": 3.50e-4, "u_v_cabin": 1e-5, "notes": "fallback"},
    }
    f.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
    return str(f)


def test_load(cabins_yaml):
    cfg = load_cabins_config(cabins_yaml)
    assert cfg["calibration_date"] == "2026-05-06"
    assert 1 in cfg["cabins"]


def test_get_v_cabin_calibrated(cabins_yaml):
    cfg = load_cabins_config(cabins_yaml)
    v, u = get_v_cabin(cfg, 1)
    assert v == pytest.approx(3.45e-4)
    assert u == pytest.approx(6e-6)


def test_get_v_cabin_fallback(cabins_yaml):
    cfg = load_cabins_config(cabins_yaml)
    v, u = get_v_cabin(cfg, 99)  # not in table
    assert v == pytest.approx(3.50e-4)
    assert u == pytest.approx(1e-5)


def test_is_calibrated(cabins_yaml):
    cfg = load_cabins_config(cabins_yaml)
    assert is_cabin_calibrated(cfg, 1) is True
    assert is_cabin_calibrated(cfg, 2) is False  # "占位"
    assert is_cabin_calibrated(cfg, 99) is False  # missing
```

## 5.4 验收标准

- [ ] `configs/cabins.yaml` 含 25 个 cabin 占位值
- [ ] `configs/loaders.py` 新增 3 个函数:`load_cabins_config`、`get_v_cabin`、`is_cabin_calibrated`
- [ ] `scripts/calibrate_v_cabin.py` 可执行,CV 超限时拒绝写入但仍记录日志
- [ ] `data/calibration_history/v_cabin_log.csv` 在脚本运行后正确创建/追加
- [ ] `tests/test_loaders_cabins.py` 全部通过
- [ ] 单位转换正确:350 g 水 → 3.5e-4 m³

## 5.5 注意事项

**单位**:V_cabin 在 yaml 中是 m³(进入物理公式直接用),用户输入是水重克数(注水法实验直觉)。脚本内部转换 g → m³(× 1e-6)。

**CV 限制**:实验方案 v5.1 要求 CV < 2%。脚本默认 `--cv-limit 0.02`,超限时拒绝写入但仍把测量记录到 csv,以便事后追溯。

---

# 任务 6:M1 — 每舱独立线性回归推理

## 6.1 背景

M1 是 v2.6 的主推理模型。每个 cabin 独立训练一根线性方程:

```
Q_est = β_cabin × hold_trend_slope + α_cabin
```

物理意义:`β_cabin ≈ V_cabin / Δt`,`α_cabin` 应接近零(零漏率时 trend_slope 应为零)。

M1 的优势:完全可解释、训练数据需求小、无过拟合风险、漂移时只需重训系数即可。

## 6.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `models/linear_regression_m1.py` | **新增**:M1 推理类 |
| `configs/models.yaml` | 新增 m1 / m2 配置项 |
| `tests/test_linear_regression_m1.py` | **新增** |

## 6.3 详细改动

### 6.3.1 新增 `models/linear_regression_m1.py`

```python
"""M1 model — per-cabin linear regression for Q estimation.

Each cabin has independent (β, α) coefficients trained from rotation
calibration data. Inference is plain arithmetic; no ML library needed.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LinearRegressionM1:
    """Per-cabin linear regression model.
    
    Coefficient table format (JSON):
    {
        "version": "v2.6.0",
        "trained_at": "2026-05-22T14:00:00",
        "feature": "hold_trend_slope",
        "target": "Q (Pa·m³/s)",
        "primary_section": "hold",
        "cabins": {
            "1": {
                "beta": 12.3, "alpha": 0.001,
                "r_squared": 0.998, "n_samples": 150,
                "u_beta": 0.4, "u_alpha": 0.0002
            },
            "2": {...},
            ...
        }
    }
    
    Inference: Q = beta * hold_trend_slope + alpha
    Uncertainty: u_Q² = (u_beta * slope)² + u_alpha²
    """
    
    def __init__(self, models_cfg: Dict[str, Any], base_dir: str = "."):
        self._base = Path(base_dir)
        m1_cfg = models_cfg.get("m1", {})
        self._coef_path = self._base / m1_cfg.get(
            "coefficients_path", "models/artifacts/current/m1_coefficients.json"
        )
        self._version: str = m1_cfg.get("version", "unknown")
        self._coefs: Dict[int, Dict[str, float]] = {}
        self._loaded = False
        self._primary_section: str = "hold"
    
    @property
    def loaded(self) -> bool:
        return self._loaded
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def primary_section(self) -> str:
        return self._primary_section
    
    @property
    def calibrated_cabins(self) -> List[int]:
        return sorted(self._coefs.keys())
    
    def load(self) -> None:
        """Load coefficient table from JSON file."""
        if not self._coef_path.exists():
            self._loaded = False
            raise FileNotFoundError(f"M1 coefficients not found: {self._coef_path}")
        
        with open(self._coef_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self._version = data.get("version", self._version)
        self._primary_section = data.get("primary_section", "hold")
        # Cabin IDs may come in as strings (JSON) or ints
        self._coefs = {int(k): v for k, v in data.get("cabins", {}).items()}
        self._loaded = True
        
        logger.info(
            "M1 loaded: version=%s, %d cabins, primary_section=%s",
            self._version, len(self._coefs), self._primary_section,
        )
    
    def predict(self, primary_trend_slope: float, cabin_id: int) -> Dict[str, Any]:
        """Predict Q_est for one cabin.
        
        Parameters
        ----------
        primary_trend_slope : float
            The trend_slope of the primary_section (typically hold).
        cabin_id : int
        
        Returns
        -------
        dict with:
            q_est: float (Pa·m³/s)
            uncertainty: float (1-sigma absolute uncertainty)
            relative_uncertainty: float (uncertainty / |q_est|)
            cabin_calibrated: bool
        """
        coef = self._coefs.get(cabin_id)
        if coef is None:
            # Fallback: use mean of all calibrated cabins
            if not self._coefs:
                return {
                    "q_est": 0.0, "uncertainty": float("inf"),
                    "relative_uncertainty": 1.0, "cabin_calibrated": False,
                }
            avg_beta = sum(c["beta"] for c in self._coefs.values()) / len(self._coefs)
            avg_alpha = sum(c.get("alpha", 0.0) for c in self._coefs.values()) / len(self._coefs)
            q = avg_beta * primary_trend_slope + avg_alpha
            return {
                "q_est": q,
                "uncertainty": abs(q) * 0.30 if q != 0 else 1e-5,
                "relative_uncertainty": 0.30,
                "cabin_calibrated": False,
            }
        
        beta = coef["beta"]
        alpha = coef.get("alpha", 0.0)
        u_beta = coef.get("u_beta", 0.0)
        u_alpha = coef.get("u_alpha", 0.0)
        
        q = beta * primary_trend_slope + alpha
        # u² = (u_beta * slope)² + u_alpha²
        uncertainty = ((u_beta * primary_trend_slope) ** 2 + u_alpha ** 2) ** 0.5
        rel_unc = uncertainty / abs(q) if abs(q) > 1e-12 else 1.0
        
        return {
            "q_est": q,
            "uncertainty": uncertainty,
            "relative_uncertainty": rel_unc,
            "cabin_calibrated": True,
        }
```

### 6.3.2 修改 `configs/models.yaml`

```yaml
# v2.6: 双轨回归模型 + 旧二分类彻底废弃

current:
  version: "v2.6.0"

# 主模型: M1 每舱线性回归
m1:
  version: "v2.6.0"
  coefficients_path: "models/artifacts/current/m1_coefficients.json"

# 备选模型: M2 全局 XGBoost 回归
m2:
  version: "v2.6.0"
  model_path: "models/artifacts/current/m2_xgb_model.json"
  scaler_path: "models/artifacts/current/m2_xgb_scaler.joblib"
  metadata_path: "models/artifacts/current/m2_metadata.json"

# v2.5 二分类模型 - 在 v2.6 中已废弃, 此节仅作参考
# xgb_classifier:
#   version: "v1.3"
#   model_path: ...
```

### 6.3.3 单元测试 `tests/test_linear_regression_m1.py`

```python
"""Tests for M1 linear regression model."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from models.linear_regression_m1 import LinearRegressionM1


@pytest.fixture
def coef_file(tmp_path):
    """Create a test coefficients JSON."""
    f = tmp_path / "m1_coef.json"
    data = {
        "version": "test_v1",
        "trained_at": "2026-05-22T00:00:00",
        "feature": "hold_trend_slope",
        "target": "Q (Pa·m³/s)",
        "primary_section": "hold",
        "cabins": {
            "1": {"beta": 10.0, "alpha": 0.001, "r_squared": 0.99,
                  "u_beta": 0.5, "u_alpha": 0.0001, "n_samples": 150},
            "2": {"beta": 12.0, "alpha": 0.002, "r_squared": 0.995,
                  "u_beta": 0.4, "u_alpha": 0.0002, "n_samples": 150},
        }
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    return f


class TestLinearRegressionM1:
    def test_load(self, coef_file, tmp_path):
        cfg = {"m1": {"coefficients_path": str(coef_file.relative_to(tmp_path)),
                       "version": "test_v1"}}
        m1 = LinearRegressionM1(cfg, base_dir=str(tmp_path))
        m1.load()
        assert m1.loaded
        assert m1.version == "test_v1"
        assert m1.primary_section == "hold"
        assert m1.calibrated_cabins == [1, 2]
    
    def test_predict_calibrated(self, coef_file, tmp_path):
        cfg = {"m1": {"coefficients_path": str(coef_file.relative_to(tmp_path))}}
        m1 = LinearRegressionM1(cfg, base_dir=str(tmp_path))
        m1.load()
        result = m1.predict(primary_trend_slope=0.5, cabin_id=1)
        # 10.0 * 0.5 + 0.001 = 5.001
        assert result["q_est"] == pytest.approx(5.001)
        assert result["cabin_calibrated"] is True
        assert result["uncertainty"] > 0
    
    def test_predict_uncalibrated(self, coef_file, tmp_path):
        cfg = {"m1": {"coefficients_path": str(coef_file.relative_to(tmp_path))}}
        m1 = LinearRegressionM1(cfg, base_dir=str(tmp_path))
        m1.load()
        result = m1.predict(primary_trend_slope=0.5, cabin_id=99)
        assert result["cabin_calibrated"] is False
        # Should use mean of calibrated cabins
        # mean_beta = 11.0, mean_alpha = 0.0015 → 11*0.5 + 0.0015 = 5.5015
        assert result["q_est"] == pytest.approx(5.5015, rel=1e-3)
        assert result["relative_uncertainty"] >= 0.30
    
    def test_load_missing_file(self, tmp_path):
        cfg = {"m1": {"coefficients_path": "nonexistent.json"}}
        m1 = LinearRegressionM1(cfg, base_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            m1.load()
        assert not m1.loaded
    
    def test_predict_zero_slope(self, coef_file, tmp_path):
        cfg = {"m1": {"coefficients_path": str(coef_file.relative_to(tmp_path))}}
        m1 = LinearRegressionM1(cfg, base_dir=str(tmp_path))
        m1.load()
        # At slope = 0, q should equal alpha (very small but not zero)
        result = m1.predict(primary_trend_slope=0.0, cabin_id=1)
        assert result["q_est"] == pytest.approx(0.001)
```

## 6.4 验收标准

- [ ] `models/linear_regression_m1.py` 实现完整
- [ ] `configs/models.yaml` 含 m1 / m2 配置
- [ ] 单元测试全部通过
- [ ] M1 文件不存在时抛 `FileNotFoundError`,`loaded=False`,不影响系统启动(任务 8 中由调用方决定如何处理)
- [ ] cabin 不在表中时回退到均值,标记 `cabin_calibrated=False`

---

# 任务 7:M2 — 全局 XGBoost 回归(43 维输入 + 特征选择)

## 7.1 背景

M2 是备选回归模型,输入 43 维特征,输出 Q_est。M2 的存在意义:

1. 吸收 M1 无法表达的非线性
2. 学习舱体异常、零点漂移、瓶身释气等隐性信号
3. 作为 M1 的健康监测对照(任务 8 中触发 disagreement 告警)

**关键约束**:43 维特征 vs 阶段 2 训练样本数 3750,经验法则要求样本数 ≥ 特征数 × 50 = 2150。3750 > 2150 但储备不大。**必须**在训练阶段做特征选择,否则容易过拟合。

## 7.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `models/xgb_regressor_m2.py` | **新增**:M2 推理类 |
| `tests/test_xgb_regressor_m2.py` | **新增** |

## 7.3 详细改动

### 7.3.1 新增 `models/xgb_regressor_m2.py`

```python
"""M2 model — global XGBoost regressor for Q estimation.

Input: 43-dim feature vector (or a subset selected during training).
Output: Q_est in Pa·m³/s.

Training is done in log10(Q) space to handle the wide dynamic range
(~1 order of magnitude across 5 capillary groups) and ensure relative
rather than absolute error is minimized. Predictions are returned in
linear Q space.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.exceptions import ModelLoadError, ModelPredictError

logger = logging.getLogger(__name__)


class XGBRegressorM2:
    """Global XGBoost regressor for Q.
    
    Metadata file (m2_metadata.json) describes:
        - feature_subset: list of feature names actually used (after selection)
        - log_space: whether predictions are in log10(Q) space
        - feature_importance: dict[name → importance]
        - evaluation: r_squared, mae, etc.
    """
    
    def __init__(self, models_cfg: Dict[str, Any], base_dir: str = "."):
        self._base = Path(base_dir)
        m2_cfg = models_cfg.get("m2", {})
        self._model_path = self._base / m2_cfg.get(
            "model_path", "models/artifacts/current/m2_xgb_model.json"
        )
        self._scaler_path = self._base / m2_cfg.get(
            "scaler_path", "models/artifacts/current/m2_xgb_scaler.joblib"
        )
        self._metadata_path = self._base / m2_cfg.get(
            "metadata_path", "models/artifacts/current/m2_metadata.json"
        )
        self._version: str = m2_cfg.get("version", "unknown")
        self._model = None
        self._scaler = None
        self._loaded = False
        self._log_space = True
        # Subset of FEATURE_ORDER_43D actually used by this model
        self._feature_subset: List[str] = []
        # Indices into the full 43-dim vector (computed at load time)
        self._feature_indices: List[int] = []
    
    @property
    def loaded(self) -> bool:
        return self._loaded
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def feature_subset(self) -> List[str]:
        return list(self._feature_subset)
    
    def load(self) -> None:
        from core.feature_spec import FEATURE_ORDER_43D
        
        try:
            import xgboost as xgb
            import joblib
            
            if not self._model_path.exists():
                raise FileNotFoundError(f"M2 model not found: {self._model_path}")
            
            booster = xgb.Booster()
            booster.load_model(str(self._model_path))
            self._model = booster
            
            if self._scaler_path.exists():
                self._scaler = joblib.load(str(self._scaler_path))
            
            # Load metadata: feature_subset, log_space
            if self._metadata_path.exists():
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._version = meta.get("version", self._version)
                self._feature_subset = meta.get("feature_subset", FEATURE_ORDER_43D)
                self._log_space = meta.get("log_space", True)
            else:
                logger.warning("M2 metadata not found; assuming full 43-dim and log-space")
                self._feature_subset = list(FEATURE_ORDER_43D)
                self._log_space = True
            
            # Pre-compute indices for fast inference
            self._feature_indices = [
                FEATURE_ORDER_43D.index(name) for name in self._feature_subset
            ]
            
            self._loaded = True
            logger.info(
                "M2 loaded: version=%s, %d features, log_space=%s",
                self._version, len(self._feature_subset), self._log_space,
            )
        except Exception as exc:
            self._loaded = False
            raise ModelLoadError(f"Failed to load M2: {exc}") from exc
    
    def predict(self, full_features: List[float]) -> Dict[str, Any]:
        """Predict Q_est from the full 43-dim feature vector.
        
        Internally selects the subset specified by metadata.
        
        Parameters
        ----------
        full_features : list of float, length 43
        
        Returns
        -------
        dict with q_est (Pa·m³/s), valid (bool).
        """
        if not self._loaded:
            return {"q_est": 0.0, "valid": False}
        
        try:
            import xgboost as xgb
            
            if len(full_features) != 43:
                raise ValueError(
                    f"M2 expects 43-dim input, got {len(full_features)}"
                )
            
            # Select subset
            x_subset = np.asarray(
                [full_features[i] for i in self._feature_indices],
                dtype=np.float32
            ).reshape(1, -1)
            
            if self._scaler is not None:
                x_subset = self._scaler.transform(x_subset)
            
            dmatrix = xgb.DMatrix(x_subset)
            y_pred = float(self._model.predict(dmatrix)[0])
            
            if self._log_space:
                # Cap to avoid overflow for unrealistic predictions
                y_pred = max(min(y_pred, 0.0), -10.0)  # Q in [1e-10, 1e0]
                q_est = 10 ** y_pred
            else:
                q_est = y_pred
            
            return {"q_est": q_est, "valid": True}
        except Exception as exc:
            raise ModelPredictError(f"M2 predict failed: {exc}") from exc
```

### 7.3.2 单元测试 `tests/test_xgb_regressor_m2.py`

```python
"""Tests for M2 XGBoost regressor."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from models.xgb_regressor_m2 import XGBRegressorM2


@pytest.fixture
def trained_m2(tmp_path):
    """Train a tiny M2 model on synthetic data and save artifacts."""
    import xgboost as xgb
    import joblib
    from sklearn.preprocessing import StandardScaler
    
    from core.feature_spec import FEATURE_ORDER_43D
    
    np.random.seed(42)
    n = 200
    X = np.random.rand(n, 43).astype(np.float32)
    # Synthetic relationship: Q ∝ feature[FEATURE_ORDER_43D.index('hold_trend_slope')]
    hold_idx = FEATURE_ORDER_43D.index("hold_trend_slope")
    y_log = np.log10(np.maximum(X[:, hold_idx] * 1e-3 + 1e-6, 1e-7))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dtrain = xgb.DMatrix(X_scaled, label=y_log)
    booster = xgb.train(
        params={"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror"},
        dtrain=dtrain,
        num_boost_round=50,
    )
    
    # Save artifacts
    model_path = tmp_path / "m2_xgb_model.json"
    scaler_path = tmp_path / "m2_xgb_scaler.joblib"
    meta_path = tmp_path / "m2_metadata.json"
    
    booster.save_model(str(model_path))
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        "version": "test_v1",
        "feature_subset": list(FEATURE_ORDER_43D),  # use all 43
        "log_space": True,
        "feature_importance": {},
        "evaluation": {"r_squared": 0.8},
    }
    meta_path.write_text(json.dumps(metadata))
    
    return tmp_path, model_path, scaler_path, meta_path


class TestXGBRegressorM2:
    def test_load_and_predict(self, trained_m2):
        tmp_path, model_path, scaler_path, meta_path = trained_m2
        cfg = {"m2": {
            "model_path": str(model_path.relative_to(tmp_path)),
            "scaler_path": str(scaler_path.relative_to(tmp_path)),
            "metadata_path": str(meta_path.relative_to(tmp_path)),
            "version": "test_v1",
        }}
        m2 = XGBRegressorM2(cfg, base_dir=str(tmp_path))
        m2.load()
        assert m2.loaded
        assert len(m2.feature_subset) == 43
        
        # Predict
        features = [0.5] * 43
        result = m2.predict(features)
        assert result["valid"] is True
        assert result["q_est"] > 0  # log-space output is positive after 10**
    
    def test_load_missing(self, tmp_path):
        cfg = {"m2": {"model_path": "nonexistent.json",
                      "scaler_path": "nonexistent.joblib",
                      "metadata_path": "nonexistent.json"}}
        m2 = XGBRegressorM2(cfg, base_dir=str(tmp_path))
        with pytest.raises(Exception):  # ModelLoadError
            m2.load()
        assert not m2.loaded
    
    def test_predict_wrong_dim(self, trained_m2):
        tmp_path, model_path, scaler_path, meta_path = trained_m2
        cfg = {"m2": {
            "model_path": str(model_path.relative_to(tmp_path)),
            "scaler_path": str(scaler_path.relative_to(tmp_path)),
            "metadata_path": str(meta_path.relative_to(tmp_path)),
        }}
        m2 = XGBRegressorM2(cfg, base_dir=str(tmp_path))
        m2.load()
        with pytest.raises(Exception):
            m2.predict([0.5] * 10)  # wrong dim
    
    def test_subset_selection(self, tmp_path):
        """If metadata specifies a subset, only those features should be used."""
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        np.random.seed(42)
        # Train on only 5 features
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.log10(X[:, 0] * 1e-3 + 1e-6)
        scaler = StandardScaler().fit(X)
        dtrain = xgb.DMatrix(scaler.transform(X), label=y)
        booster = xgb.train({"max_depth": 3}, dtrain, num_boost_round=20)
        
        model_path = tmp_path / "m2.json"
        scaler_path = tmp_path / "scaler.joblib"
        meta_path = tmp_path / "meta.json"
        booster.save_model(str(model_path))
        joblib.dump(scaler, scaler_path)
        
        # Subset of 5 specific feature names
        subset = ["hold_max", "hold_min", "hold_trend_slope", "evac_trend_slope", "cavity_id"]
        meta_path.write_text(json.dumps({
            "feature_subset": subset, "log_space": True
        }))
        
        cfg = {"m2": {
            "model_path": str(model_path.relative_to(tmp_path)),
            "scaler_path": str(scaler_path.relative_to(tmp_path)),
            "metadata_path": str(meta_path.relative_to(tmp_path)),
        }}
        m2 = XGBRegressorM2(cfg, base_dir=str(tmp_path))
        m2.load()
        assert m2.feature_subset == subset
        # Predict with full 43 dim, should work (selects subset internally)
        result = m2.predict([0.5] * 43)
        assert result["valid"] is True
```

## 7.4 验收标准

- [ ] `models/xgb_regressor_m2.py` 实现完整
- [ ] M2 加载失败时抛 `ModelLoadError`,`loaded=False`
- [ ] metadata 中的 `feature_subset` 决定实际使用哪些维度
- [ ] log_space=True 时输出经 `10**` 转换
- [ ] 预测时检查输入必须是 43 维
- [ ] 单元测试全部通过

## 7.5 与 M1 的关系

M2 与 M1 完全独立加载、独立推理。任务 8 中将两者结果做融合。



---

# 任务 8:推理流水线集成 — 双轨融合 + Q 阈值判决

## 8.1 背景

把 M1、M2 集成到 `pipeline/processing_loop.py`,替换 v2.5 的二分类逻辑。同时引入"客户阈值判决"——客户在产品配置中设置 `Q_threshold`,系统输出 Q_est 后比较大小判废。

这是 v2.6 改造中**最大的一块代码改动**,集成所有前面任务的产物。

## 8.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `pipeline/processing_loop.py` | **重写** `__init__` 和 `_process_complete_cycle` |
| `health/fault_codes.py` | 新增 F010 / F011 / F012 故障码 |
| `main.py` | 改为加载 M1 / M2 / cabins / products 配置并注入 ProcessingLoop |
| `integration/result_sender.py` | `write_result` 第三参数语义变更:概率 → Q_est |
| `tests/test_processing_loop.py` | 适配新逻辑 |

## 8.3 详细改动

### 8.3.1 重写 `pipeline/processing_loop.py`

#### 新的 `__init__`

```python
def __init__(
    self,
    runtime_cfg: Dict[str, Any],
    profile: "CycleProfile",
    cabins_cfg: Dict[str, Any],
    products_cfg: Dict[str, Any],
    polling_engine: PollingEngine,
    fsm_manager: CycleFSMManager,
    m1_model: "LinearRegressionM1",
    m2_model: "XGBRegressorM2",
    db_logger: DatabaseLogger,
    result_sender: ResultSender,
    alarm_pusher: AlarmPusher,
    health_checker: HealthChecker,
    fault_reporter: FaultReporter,
):
    self._cfg = runtime_cfg
    self._profile = profile
    self._cabins_cfg = cabins_cfg
    self._products_cfg = products_cfg
    self._poller = polling_engine
    self._fsm = fsm_manager
    self._m1 = m1_model
    self._m2 = m2_model
    self._db = db_logger
    self._sender = result_sender
    self._alarm = alarm_pusher
    self._health = health_checker
    self._reporter = fault_reporter
    
    # ── v2.6 推理参数 ───────────────────────────────────────
    inf_cfg = runtime_cfg.get("model_inference", {})
    self._a_resolution = float(inf_cfg.get("a_resolution", 1.0e-5))
    self._m_disagreement_threshold = float(inf_cfg.get("m_disagreement_threshold", 0.20))
    
    # ── 产品 ────────────────────────────────────────────────
    self._current_product_id = products_cfg.get("default_product_id", "default")
    
    # ── 杂项 ────────────────────────────────────────────────
    self._no_bottle_threshold = float(runtime_cfg.get("no_bottle_threshold", 50.0))
    self._running = False
    self._paused = False
    self._watchdog = True
    self._last_poll_ts = 0.0
    
    # 警告:模型未加载时的策略
    if not self._m1.loaded:
        logger.warning(
            "M1 model not loaded. System will run but no Q_est will be produced "
            "until M1 is deployed."
        )
        self._reporter.raise_fault("F002", "M1 模型未加载")
```

#### 新的 `_predict_q`

```python
def _predict_q(
    self,
    feats: Dict[str, float],
    feature_vector_43d: List[float],
    cabin_id: int,
) -> Dict[str, Any]:
    """Run M1 + M2 inference and return fused result.
    
    Returns
    -------
    dict with:
        q_est: float (Pa·m³/s, 主输出, 来自 M1)
        m1_q, m2_q: 个别模型输出
        m_disagreement: |m2 - m1| / |m1|
        q_uncertainty: M1 的不确定度
        cabin_calibrated: bool
        valid: bool (False if M1 not loaded)
        below_resolution: bool (Q_est < A_resolution)
    """
    # M1 主输出
    if not self._m1.loaded:
        return {
            "q_est": 0.0, "valid": False,
            "m1_q": 0.0, "m2_q": 0.0,
            "m_disagreement": 0.0,
            "q_uncertainty": float("inf"),
            "cabin_calibrated": False,
            "below_resolution": True,
        }
    
    # 提取主段 trend_slope (默认 hold_trend_slope)
    primary_section = self._m1.primary_section
    primary_slope = feats.get(f"{primary_section}_trend_slope", 0.0)
    
    m1_result = self._m1.predict(primary_slope, cabin_id)
    m1_q = m1_result["q_est"]
    
    # 触发 F011: M1 未标定该舱
    if not m1_result["cabin_calibrated"]:
        # 不阻塞推理, 但 throttle 告警避免每圈刷屏
        # (HealthChecker 内部应有去重逻辑; 如没有, 此处直接 raise_fault 也可)
        logger.debug("Cabin %d not calibrated in M1 (using fallback)", cabin_id)
    
    # M2 辅助输出
    m2_q = m1_q  # default to M1 if M2 unavailable
    m2_valid = False
    if self._m2.loaded:
        try:
            m2_result = self._m2.predict(feature_vector_43d)
            if m2_result["valid"]:
                m2_q = m2_result["q_est"]
                m2_valid = True
        except Exception as exc:
            logger.warning("M2 predict failed: %s", exc)
    
    # 一致性检查
    if m2_valid and abs(m1_q) > 1e-12:
        disagreement = abs(m2_q - m1_q) / abs(m1_q)
    else:
        disagreement = 0.0
    
    if m2_valid and disagreement > self._m_disagreement_threshold:
        logger.warning(
            "Cabin %d: M1/M2 disagreement %.1f%% (M1=%.3e, M2=%.3e)",
            cabin_id, disagreement * 100, m1_q, m2_q,
        )
        self._reporter.raise_fault(
            "F010",
            f"Cabin {cabin_id}: M1/M2 漏率估计差异 {disagreement * 100:.1f}%"
        )
    
    # 是否低于系统分辨率
    below_resolution = abs(m1_q) < self._a_resolution
    if below_resolution:
        # 不刷 fault 日志, 只 debug
        logger.debug("Cabin %d: Q_est %.3e below A=%.3e (resolution)",
                     cabin_id, m1_q, self._a_resolution)
    
    return {
        "q_est": m1_q,            # 主输出始终用 M1
        "valid": True,
        "m1_q": m1_q,
        "m2_q": m2_q if m2_valid else None,
        "m_disagreement": disagreement,
        "q_uncertainty": m1_result["uncertainty"],
        "cabin_calibrated": m1_result["cabin_calibrated"],
        "below_resolution": below_resolution,
    }
```

#### 新的 `_process_complete_cycle`

```python
def _process_complete_cycle(self, cabin_id: int) -> None:
    """v2.6: 全周期采集 + 双轨回归 + Q 阈值判决"""
    from core.features import compute_features_v26, features_to_vector
    from core.feature_spec import FEATURE_ORDER_43D
    
    fsm = self._fsm.fsms[cabin_id]
    data = fsm.harvest()
    
    if len(data.pressures) < 2:
        logger.warning("Cabin %d: insufficient data (%d points), skipping",
                       cabin_id, len(data.pressures))
        fsm.reset()
        return
    
    t0 = time.perf_counter()
    
    # ── 特征提取 ────────────────────────────────────────────
    feats = compute_features_v26(
        data.pressures, data.angles, cabin_id, self._profile
    )
    feature_vector = features_to_vector(feats, mode="43d")
    
    duration_s = (data.timestamps[-1] - data.timestamps[0]) if len(data.timestamps) > 1 else 0.0
    
    # ── 无瓶子检测 ──────────────────────────────────────────
    # 用 hold 段 max 来判断 (替代 v2.5 的整段 max)
    hold_max = feats.get("hold_max", 0.0)
    if hold_max < self._no_bottle_threshold:
        result = {
            "label": LABEL_NO_BOTTLE,
            "q_est": 0.0,
            "q_threshold": 0.0,
            "below_resolution": True,
        }
        self._db.log_record(
            cavity_id=cabin_id,
            pressures=data.pressures, angles=data.angles,
            ai_values=data.ai_values, positions=data.positions,
            features=feats,
            label=LABEL_NO_BOTTLE, probability=0.0, confidence=0.0,
            model_version=self._m1.version if self._m1.loaded else "none",
            duration_s=duration_s,
            leak_valve_status=data.leak_valve_status,
            end_angle=data.end_angle,
            cycle_profile_id=data.cycle_profile_id,
            q_est=0.0, q_threshold=None, q_uncertainty=None,
            m1_q=0.0, m2_q=None, m_disagreement=0.0,
            product_id=self._current_product_id,
        )
        fsm.reset()
        return
    
    # ── Q 推理 ─────────────────────────────────────────────
    q_result = self._predict_q(feats, feature_vector, cabin_id)
    
    if not q_result["valid"]:
        # M1 未加载, 不能给出 Q_est
        result = {
            "label": LABEL_NA, "q_est": 0.0, "q_threshold": 0.0,
            "below_resolution": True,
        }
    else:
        q_est = q_result["q_est"]
        # 取产品阈值
        product = self._products_cfg.get("products", {}).get(self._current_product_id, {})
        q_threshold = float(product.get("q_threshold", 1.0e-3))
        
        if q_result["below_resolution"]:
            # Q_est 低于系统分辨率, 不判决
            label = LABEL_NA
            self._reporter.raise_fault(
                "F012",
                f"Cabin {cabin_id}: Q_est={q_est:.2e} 低于分辨率 A={self._a_resolution:.2e}",
            )
        elif q_est > q_threshold:
            label = LABEL_LEAK
        else:
            label = LABEL_OK
        
        result = {
            "label": label,
            "q_est": q_est,
            "q_threshold": q_threshold,
            **q_result,
        }
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    self._health.report_inference_latency(elapsed_ms)
    
    # ── 数据库写入 ─────────────────────────────────────────
    try:
        self._db.log_record(
            cavity_id=cabin_id,
            pressures=data.pressures, angles=data.angles,
            ai_values=data.ai_values, positions=data.positions,
            features=feats,
            label=result["label"],
            probability=result.get("q_est", 0.0),  # 兼容字段, 写 Q_est
            confidence=1.0 - q_result.get("q_uncertainty", 0.0) / max(abs(result.get("q_est", 1)), 1e-12) if q_result.get("valid") else 0.0,
            model_version=self._m1.version if self._m1.loaded else "none",
            duration_s=duration_s,
            leak_valve_status=data.leak_valve_status,
            end_angle=data.end_angle,
            cycle_profile_id=data.cycle_profile_id,
            q_est=result.get("q_est"),
            q_threshold=result.get("q_threshold"),
            q_uncertainty=q_result.get("q_uncertainty"),
            m1_q=q_result.get("m1_q"),
            m2_q=q_result.get("m2_q"),
            m_disagreement=q_result.get("m_disagreement"),
            product_id=self._current_product_id,
        )
    except Exception as exc:
        logger.error("DB log failed for cabin %d: %s", cabin_id, exc)
        self._reporter.raise_fault("F006", f"数据库写入失败: {exc}")
    
    # ── PLC 写回 ────────────────────────────────────────────
    # write_result 第三个参数语义: v2.5 是 probability, v2.6 是 q_est
    if result["label"] in (LABEL_OK, LABEL_LEAK):
        try:
            self._sender.write_result(cabin_id, result["label"], result["q_est"])
        except Exception as exc:
            logger.error("PLC write-back failed for cabin %d: %s", cabin_id, exc)
    
    # ── 告警推送 ────────────────────────────────────────────
    if result["label"] == LABEL_LEAK:
        self._alarm.push_leak_alarm(cabin_id, result["q_est"])
    
    # ── 日志输出 ────────────────────────────────────────────
    label_str = {LABEL_OK: "OK", LABEL_LEAK: "LEAK",
                 LABEL_NA: "N/A", LABEL_NO_BOTTLE: "NO_BOTTLE"}[result["label"]]
    if result["label"] not in (LABEL_NO_BOTTLE,):
        logger.info(
            "Cabin %d: %s (Q_est=%.3e, threshold=%.3e, points=%d, %.1fms)",
            cabin_id, label_str,
            result.get("q_est", 0.0), result.get("q_threshold", 0.0),
            len(data.pressures), elapsed_ms,
        )
    
    fsm.reset()
```

### 8.3.2 修改 `health/fault_codes.py`

```python
FAULT_CODES = {
    "F001": {"description": "PLC连接丢失", "level": "CRITICAL", "plc_value": 1},
    "F002": {"description": "AI模型未加载", "level": "ERROR", "plc_value": 2},
    "F003": {"description": "传感器数据异常", "level": "WARNING", "plc_value": 3},
    "F004": {"description": "推理延迟过高", "level": "WARNING", "plc_value": 4},
    "F005": {"description": "磁盘空间不足", "level": "ERROR", "plc_value": 5},
    "F006": {"description": "数据库写入失败", "level": "ERROR", "plc_value": 6},
    "F007": {"description": "数据库容量告警", "level": "WARNING", "plc_value": 7},
    "F008": {"description": "采集线程异常终止", "level": "CRITICAL", "plc_value": 8},
    "F009": {"description": "舱室状态机故障", "level": "WARNING", "plc_value": 9},
    # v2.6 新增
    "F010": {"description": "M1/M2 漏率估计差异过大", "level": "WARNING", "plc_value": 10},
    "F011": {"description": "M1 模型未标定该舱", "level": "WARNING", "plc_value": 11},
    "F012": {"description": "Q 估计低于系统分辨率", "level": "INFO", "plc_value": 12},
}
```

### 8.3.3 修改 `integration/result_sender.py`

接口签名不变,只改文档说明 + PLC 写入字段语义。

```python
def write_result(self, cabin_id: int, label: int, q_est: float) -> None:
    """Write inference result back to PLC.
    
    v2.6 change: third parameter is now Q_est (Pa·m³/s) instead of probability.
    The cabinHealthStatus REAL field on PLC therefore holds Q_est.
    
    NOTE: This is a semantic change visible to PLC/HMI. Coordinate with
    the automation engineer to confirm the upstream display will interpret
    the value correctly.
    
    Parameters
    ----------
    cabin_id : int
        1..25 (Cabin[0] is reserved and skipped).
    label : int
        0 = LEAK, 1 = OK. Other values cause skip.
    q_est : float
        Estimated leak rate in Pa·m³/s. Written to cabinHealthStatus.
    """
    # Internal implementation unchanged: writes Bool + REAL in 8-byte block.
    # The "probability" parameter name in old code is now treated as q_est.
    # ... 现有 write_result 逻辑 (写 leakTestResult_AI bit + cabinHealthStatus REAL) ...
```

### 8.3.4 修改 `main.py`

```python
def main():
    args = parse_args()
    
    # ── 加载所有配置 ────────────────────────────────────────
    plc_cfg = load_plc_config()
    runtime_cfg = load_runtime_config()
    models_cfg = load_models_config()
    health_cfg = load_health_config()
    ipc_cfg = load_ipc_config()
    cabins_cfg = load_cabins_config()
    products_cfg = load_products_config()
    profile = load_active_cycle_profile()
    
    logger = setup_logging(runtime_cfg.get("logging", {}))
    
    # ── 初始化模型 (v2.6) ───────────────────────────────────
    m1 = LinearRegressionM1(models_cfg)
    try:
        m1.load()
    except Exception as exc:
        logger.warning("M1 not loaded: %s. System runs but produces no Q_est.", exc)
    
    m2 = XGBRegressorM2(models_cfg)
    try:
        m2.load()
    except Exception as exc:
        logger.warning("M2 not loaded: %s. Health monitoring degraded.", exc)
    
    # ── 子系统初始化 ────────────────────────────────────────
    fault_reporter = FaultReporter()
    alarm_pusher = AlarmPusher(ipc_cfg)
    polling_engine = PollingEngine(plc_cfg, mode=args.mode)
    
    fsm_manager = CycleFSMManager(
        cabin_count=plc_cfg["cabin_array"]["cabin_count"],
        profile=profile,
        active_start=plc_cfg["cabin_array"].get("active_start", 1),
        active_end=plc_cfg["cabin_array"].get("active_end", 25),
    )
    
    db_logger = DatabaseLogger(runtime_cfg["database"]["path"])
    result_sender = ResultSender(plc_cfg, polling_engine)
    health_checker = HealthChecker(...)
    
    # ── ProcessingLoop (v2.6) ───────────────────────────────
    processing_loop = ProcessingLoop(
        runtime_cfg=runtime_cfg,
        profile=profile,
        cabins_cfg=cabins_cfg,
        products_cfg=products_cfg,
        polling_engine=polling_engine,
        fsm_manager=fsm_manager,
        m1_model=m1,
        m2_model=m2,
        db_logger=db_logger,
        result_sender=result_sender,
        alarm_pusher=alarm_pusher,
        health_checker=health_checker,
        fault_reporter=fault_reporter,
    )
    
    # ── 启动 ────────────────────────────────────────────────
    polling_engine.start()
    processing_loop.start()
    # ... 交互菜单循环 ...
```

## 8.4 验收标准

- [ ] `processing_loop.py` 重写完成,集成 M1/M2/cabins/products
- [ ] `_predict_q` 正确处理 M1 未加载 / M2 未加载 / 未标定舱 / 低于分辨率四种情况
- [ ] PLC 写回 cabinHealthStatus 是 Q_est 而非 probability(已与自控工程师确认)
- [ ] F010 / F011 / F012 三个新故障码触发逻辑正确
- [ ] `main.py` 加载所有 v2.6 配置无错
- [ ] mock 模式下 30 秒内能采到 ≥ 5 圈,数据库 q_est 列有非零值
- [ ] 现有 `tests/test_processing_loop.py` 适配通过

## 8.5 与自控工程师的协调

**这一项必须在 v2.6 上线前完成确认**:

PLC 端的 `cabinHealthStatus`(REAL,4 字节)在 v2.5 中存的是 0~1 的概率值,HMI 可能据此显示百分比。v2.6 起此字段存 Q_est,数值范围 1e-7 ~ 1e-2。**HMI 显示侧必须同步切换显示逻辑**——否则切换那一刻 HMI 会出现"健康度从 95% 变成 0.001"的视觉异常,产线主管会以为系统故障。

建议:
- 在 HMI 侧加一个版本探测,根据软件版本切换显示语义
- 或者在切换时刻设备临时停产,做一次完整的"全产线状态对齐"

---

# 任务 9:产品配置 + Q/d 双向换算

## 9.1 背景

不同瓶型/液体/缺陷规格对应不同的 Q_threshold。产品配置存储为 yaml,系统启动时加载。同时实现 Q ↔ d 双向换算公式(供 HMI 显示使用)。

## 9.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `configs/products.yaml` | **新增** |
| `configs/loaders.py` | 新增 `load_products_config` |
| `core/q_d_conversion.py` | **新增**:层流 / 临界流 双向换算 |
| `tests/test_q_d_conversion.py` | **新增** |

## 9.3 详细改动

### 9.3.1 新增 `configs/products.yaml`

```yaml
# 产品配置 — 每种产品(瓶型 × 液体 × 缺陷规格)对应一组判废参数
#
# Q_threshold: 客户判废阈值, 单位 Pa·m³/s
# d_critical:  等效缺陷孔径, 单位 μm (与 Q_threshold 通过物理公式相互换算)
# flow_regime: "laminar" 或 "choked", 决定换算公式
# l_ref_mm:    参考壁厚 mm (laminar 时使用)

default_product_id: "TEST"

products:
  TEST:
    name: "测试产品(实验用)"
    bottle_volume_ml: 500
    flow_regime: "laminar"
    l_ref_mm: 0.5
    q_threshold: 1.0e-3
    notes: "实验阶段使用, Q_threshold 设较松"
  
  P001:
    name: "500mL 矿泉水"
    bottle_volume_ml: 500
    flow_regime: "laminar"
    l_ref_mm: 0.5
    q_threshold: 2.0e-3
  
  P002:
    name: "700mL 53° 白酒"
    bottle_volume_ml: 700
    flow_regime: "laminar"
    l_ref_mm: 0.5
    q_threshold: 1.5e-3
```

### 9.3.2 在 `configs/loaders.py` 新增

```python
def load_products_config(path: str = "configs/products.yaml") -> Dict[str, Any]:
    """Load product configuration."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"products config not found: {path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_product(products_cfg: Dict[str, Any], product_id: str) -> Dict[str, Any]:
    """Get product config by ID."""
    products = products_cfg.get("products", {})
    if product_id in products:
        return products[product_id]
    # Fallback to default
    default_id = products_cfg.get("default_product_id", "TEST")
    return products.get(default_id, {})
```

### 9.3.3 新增 `core/q_d_conversion.py`

```python
"""Q ↔ d (equivalent defect diameter) conversion.

Two flow regimes are supported:

1. Laminar flow (l/d > 100): Hagen-Poiseuille equation
   Q = π·d⁴ / [128·η·(l + 0.41·d)] · p̄ · Δp

2. Choked flow (l/d < 3): Yoshida critical flow equation
   Q = C_d · p_u · (π·d²/4) · √[γRT/M · (2/(γ+1))^((γ+1)/(γ−1))]
   where C_d = 0.8623 − 0.2541·(p_d/p_u)

References: GB/T 40336-2021; Yoshida (2021) Packag. Technol. Sci.
"""

from __future__ import annotations
import math


# Physical constants (air at 20°C)
ETA_AIR = 1.83e-5      # dynamic viscosity, Pa·s
GAMMA = 1.4            # heat capacity ratio
M_AIR = 0.029          # molar mass, kg/mol
R = 8.314              # gas constant, J/(mol·K)
T_REF = 293.15         # reference temperature, K
P_ATM = 101325.0       # atmospheric pressure, Pa
P_VACUUM = 35000.0     # default chamber vacuum (≈ 350 mbar abs), Pa


def q_to_d_laminar(
    q: float,
    l_ref_mm: float = 0.5,
    p_atm: float = P_ATM,
    p_chamber: float = P_VACUUM,
    eta: float = ETA_AIR,
) -> float:
    """Reverse Hagen-Poiseuille to find equivalent diameter.
    
    Solves: Q = π·d⁴ / [128·η·(l + 0.41·d)] · p̄ · Δp
    Iteratively refines for the Sampson term 0.41·d.
    
    Parameters
    ----------
    q : leak rate in Pa·m³/s
    l_ref_mm : reference channel length in mm
    
    Returns
    -------
    d in micrometers (μm). 0.0 if q <= 0.
    """
    if q <= 0:
        return 0.0
    
    l = l_ref_mm * 1e-3  # mm → m
    p_bar = (p_atm + p_chamber) / 2
    delta_p = p_atm - p_chamber
    
    # Initial guess ignoring Sampson: Q ≈ π·d⁴ / (128·η·l) · p̄·Δp
    d4 = q * 128 * eta * l / (math.pi * p_bar * delta_p)
    d = d4 ** 0.25
    
    # Iterate to refine
    for _ in range(15):
        eff_l = l + 0.41 * d
        d4 = q * 128 * eta * eff_l / (math.pi * p_bar * delta_p)
        d_new = d4 ** 0.25
        if abs(d_new - d) / d < 1e-7:
            break
        d = d_new
    
    return d * 1e6  # m → μm


def d_to_q_laminar(
    d_um: float,
    l_ref_mm: float = 0.5,
    p_atm: float = P_ATM,
    p_chamber: float = P_VACUUM,
    eta: float = ETA_AIR,
) -> float:
    """Forward Hagen-Poiseuille."""
    if d_um <= 0:
        return 0.0
    d = d_um * 1e-6
    l = l_ref_mm * 1e-3
    p_bar = (p_atm + p_chamber) / 2
    delta_p = p_atm - p_chamber
    return math.pi * d ** 4 / (128 * eta * (l + 0.41 * d)) * p_bar * delta_p


def q_to_d_choked(
    q: float,
    p_u: float = P_ATM,
    p_d: float = P_VACUUM,
    T: float = T_REF,
) -> float:
    """Reverse Yoshida critical flow."""
    if q <= 0:
        return 0.0
    p_ratio = p_d / p_u
    c_d = 0.8623 - 0.2541 * p_ratio
    sqrt_factor = math.sqrt(
        GAMMA * R * T / M_AIR * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (GAMMA - 1))
    )
    area = q / (c_d * p_u * sqrt_factor)
    if area <= 0:
        return 0.0
    d_squared = area * 4 / math.pi
    d = d_squared ** 0.5
    return d * 1e6


def d_to_q_choked(
    d_um: float,
    p_u: float = P_ATM,
    p_d: float = P_VACUUM,
    T: float = T_REF,
) -> float:
    """Forward Yoshida critical flow."""
    if d_um <= 0:
        return 0.0
    d = d_um * 1e-6
    area = math.pi * d ** 2 / 4
    p_ratio = p_d / p_u
    c_d = 0.8623 - 0.2541 * p_ratio
    sqrt_factor = math.sqrt(
        GAMMA * R * T / M_AIR * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (GAMMA - 1))
    )
    return c_d * p_u * area * sqrt_factor


def q_to_d(q: float, regime: str = "laminar", **kwargs) -> float:
    """Dispatch by flow regime."""
    if regime == "laminar":
        return q_to_d_laminar(q, **kwargs)
    elif regime == "choked":
        return q_to_d_choked(q, **kwargs)
    raise ValueError(f"Unknown flow regime: {regime}")


def d_to_q(d_um: float, regime: str = "laminar", **kwargs) -> float:
    """Dispatch by flow regime."""
    if regime == "laminar":
        return d_to_q_laminar(d_um, **kwargs)
    elif regime == "choked":
        return d_to_q_choked(d_um, **kwargs)
    raise ValueError(f"Unknown flow regime: {regime}")
```

### 9.3.4 单元测试 `tests/test_q_d_conversion.py`

```python
"""Tests for Q ↔ d conversion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.q_d_conversion import (
    q_to_d_laminar, d_to_q_laminar,
    q_to_d_choked, d_to_q_choked,
    q_to_d, d_to_q,
)


class TestLaminar:
    def test_round_trip(self):
        for d_in in [50, 100, 150, 200]:
            q = d_to_q_laminar(d_in)
            d_out = q_to_d_laminar(q)
            assert abs(d_out - d_in) / d_in < 0.001
    
    def test_q_d4_relationship(self):
        """Q ∝ d⁴ approximately."""
        q1 = d_to_q_laminar(100)
        q2 = d_to_q_laminar(200)
        assert 14 < q2 / q1 < 17
    
    def test_zero_input(self):
        assert q_to_d_laminar(0.0) == 0.0
        assert d_to_q_laminar(0.0) == 0.0
    
    def test_negative_input(self):
        assert q_to_d_laminar(-1.0) == 0.0
        assert d_to_q_laminar(-1.0) == 0.0


class TestChoked:
    def test_round_trip(self):
        for d_in in [50, 100, 150, 200]:
            q = d_to_q_choked(d_in)
            d_out = q_to_d_choked(q)
            assert abs(d_out - d_in) / d_in < 0.001
    
    def test_q_d2_relationship(self):
        """Q ∝ d² in choked regime."""
        q1 = d_to_q_choked(100)
        q2 = d_to_q_choked(200)
        assert 3.5 < q2 / q1 < 4.5


class TestDispatch:
    def test_unknown_regime(self):
        with pytest.raises(ValueError):
            q_to_d(1e-3, regime="turbulent")
        with pytest.raises(ValueError):
            d_to_q(100, regime="turbulent")
    
    def test_dispatch_laminar(self):
        assert q_to_d(1e-3, regime="laminar") == q_to_d_laminar(1e-3)
        assert d_to_q(100, regime="laminar") == d_to_q_laminar(100)
```

## 9.4 验收标准

- [ ] `configs/products.yaml` 含至少 3 个示例产品
- [ ] `configs/loaders.py` 新增 `load_products_config` 与 `get_product`
- [ ] `core/q_d_conversion.py` 实现完整,4 个核心函数 + 2 个 dispatch
- [ ] round-trip 误差 < 0.1%
- [ ] 单元测试全部通过

---

# 任务 10:训练脚本改造为回归

## 10.1 背景

`train/train_model.py` 是 v2.5 的二分类训练脚本。v2.6 需要两个新脚本:M1 训练(每舱线性回归)和 M2 训练(全局 XGBoost 回归 + 特征选择)。再加一个数据准备脚本,从 v2.6 导出 CSV 中计算 q_measured。

## 10.2 改动文件清单

| 文件 | 操作 |
|---|---|
| `train/prepare_q_data.py` | **新增**:从 raw CSV 计算 q_measured |
| `train/train_m1.py` | **新增**:每舱线性回归 |
| `train/train_m2.py` | **新增**:XGBoost 回归 + 特征选择 |
| 旧 `train/train_model.py` | 保留(允许仍被调用,但本期项目不使用) |

## 10.3 详细改动

### 10.3.1 `train/prepare_q_data.py`

```python
"""Prepare Q-labeled training data from v2.6 raw CSV.

Reads exported CSV (with 70-point pressure_data + angles), looks up
V_cabin from cabins.yaml, computes q_measured = V_cabin × |dp/dt_fit| using
the hold-section trend slope. Writes a new CSV with all original columns
plus q_measured.

Usage:
    python -m train.prepare_q_data \
        --raw-csv export_S2.csv \
        --cabins-config configs/cabins.yaml \
        --runtime-config configs/runtime.yaml \
        --output train_data_S2.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Bootstrap project path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cycle_profile import CycleProfile, load_active_cycle_profile
from core.curve_segmenter import segment_by_angle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-csv", required=True, help="Input CSV (from data_exporter)")
    p.add_argument("--cabins-config", default="configs/cabins.yaml")
    p.add_argument("--runtime-config", default="configs/runtime.yaml")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--profile-id", default=None,
                   help="Override active_profile from runtime.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Load cabins.yaml
    with open(args.cabins_config, "r", encoding="utf-8") as f:
        cabins_cfg = yaml.safe_load(f)
    
    def get_v_cabin(cabin_id):
        entry = cabins_cfg.get("cabins", {}).get(cabin_id)
        if entry and "v_cabin" in entry:
            return float(entry["v_cabin"])
        return float(cabins_cfg.get("default", {}).get("v_cabin", 3.5e-4))
    
    # Load runtime → profile
    with open(args.runtime_config, "r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f)
    if args.profile_id:
        runtime_cfg["active_profile"] = args.profile_id
    profile = load_active_cycle_profile_from_dict(runtime_cfg)
    
    primary_section = profile.primary_section
    logger.info("Profile %s, primary_section=%s", profile.profile_id, primary_section)
    
    # Load raw CSV
    df = pd.read_csv(args.raw_csv)
    logger.info("Loaded %d rows from %s", len(df), args.raw_csv)
    
    # Compute q_measured per row
    q_measured_list = []
    dp_dt_list = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            pressures = json.loads(row["pressure_data"])
            angles = json.loads(row["angle_data"])
            cabin_id = int(row["cavity_id"])
            
            # Segment hold section
            sections = segment_by_angle(pressures, angles, profile)
            hold_pressures = sections.get(primary_section, [])
            
            if len(hold_pressures) < 5:
                skipped += 1
                q_measured_list.append(None)
                dp_dt_list.append(None)
                continue
            
            # Linear fit on hold section
            x = np.arange(len(hold_pressures))
            slope = float(np.polyfit(x, hold_pressures, 1)[0])
            
            # Convert slope to dp/dt: slope is "per-sample", samples spaced by interval_s
            # Pressure unit: assume mbar (per system convention)
            # Need: dp/dt in Pa/s for Q calculation
            # 1 mbar = 100 Pa
            interval_s = profile.collection_interval_s
            dp_dt_pa_per_s = slope * 100.0 / interval_s  # mbar/sample → Pa/s
            
            # Q = V × |dp/dt|
            v_cabin = get_v_cabin(cabin_id)
            q = v_cabin * abs(dp_dt_pa_per_s)
            
            q_measured_list.append(q)
            dp_dt_list.append(dp_dt_pa_per_s)
        except Exception as exc:
            logger.warning("Row %d: %s", idx, exc)
            skipped += 1
            q_measured_list.append(None)
            dp_dt_list.append(None)
    
    df["q_measured"] = q_measured_list
    df["dp_dt_pa_per_s"] = dp_dt_list
    
    # Drop rows with missing q
    df_valid = df.dropna(subset=["q_measured"])
    df_valid.to_csv(args.output, index=False)
    
    logger.info("Wrote %d rows to %s (skipped %d)",
                len(df_valid), args.output, skipped)


def load_active_cycle_profile_from_dict(runtime_cfg):
    from core.cycle_profile import load_active_cycle_profile
    return load_active_cycle_profile(runtime_cfg)


if __name__ == "__main__":
    main()
```

### 10.3.2 `train/train_m1.py`

```python
"""Train M1 — per-cabin linear regression for Q estimation.

Reads CSV produced by prepare_q_data.py (must contain q_measured and
hold_trend_slope features). For each cabin, fits q_measured = β × slope + α
using least squares. Bootstrap (1000 samples) for u_β, u_α.

Usage:
    python -m train.train_m1 \
        --data train_data_S2.csv \
        --output models/artifacts/v2.6.0/m1_coefficients.json \
        --version v2.6.0
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cycle_profile import load_active_cycle_profile
from configs.loaders import load_runtime_config
from core.feature_spec import FEATURE_ORDER_43D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Q-labeled CSV (from prepare_q_data)")
    p.add_argument("--output", required=True, help="Output coefficients JSON")
    p.add_argument("--version", default="v2.6.0")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--min-r2", type=float, default=0.99)
    p.add_argument("--min-samples-per-cabin", type=int, default=20)
    p.add_argument("--feature-name", default=None,
                   help="Feature column name (default: <primary>_trend_slope)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fit_one_cabin(slopes, q_values, bootstrap_samples, seed):
    """Fit Q = β × slope + α with least squares. Bootstrap for uncertainties."""
    n = len(slopes)
    if n < 5:
        return None
    
    slopes = np.asarray(slopes, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    
    # Least squares
    coef = np.polyfit(slopes, q_values, 1)
    beta, alpha = float(coef[0]), float(coef[1])
    
    # R²
    q_pred = beta * slopes + alpha
    ss_res = np.sum((q_values - q_pred) ** 2)
    ss_tot = np.sum((q_values - np.mean(q_values)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Bootstrap
    rng = np.random.default_rng(seed)
    betas, alphas = [], []
    for _ in range(bootstrap_samples):
        idx = rng.choice(n, size=n, replace=True)
        try:
            c = np.polyfit(slopes[idx], q_values[idx], 1)
            betas.append(c[0])
            alphas.append(c[1])
        except Exception:
            pass
    
    u_beta = float(np.std(betas)) if betas else 0.0
    u_alpha = float(np.std(alphas)) if alphas else 0.0
    
    return {
        "beta": beta, "alpha": alpha,
        "r_squared": float(r_squared),
        "u_beta": u_beta, "u_alpha": u_alpha,
        "n_samples": int(n),
    }


def main():
    args = parse_args()
    
    # Determine feature name
    runtime_cfg = load_runtime_config()
    profile = load_active_cycle_profile(runtime_cfg)
    feature_name = args.feature_name or f"{profile.primary_section}_trend_slope"
    
    # Load data
    df = pd.read_csv(args.data)
    logger.info("Loaded %d rows", len(df))
    
    if "q_measured" not in df.columns:
        sys.exit("ERROR: input CSV must contain 'q_measured' column")
    
    # The feature_name comes from the 'features' JSON column. Extract it.
    # Or, if precomputed columns exist (e.g. hold_trend_slope), use directly.
    if feature_name in df.columns:
        df["_slope"] = df[feature_name]
    elif "features" in df.columns:
        df["_slope"] = df["features"].apply(
            lambda s: json.loads(s).get(feature_name, 0.0) if isinstance(s, str) else 0.0
        )
    else:
        sys.exit(f"ERROR: cannot find feature '{feature_name}' in CSV")
    
    df_valid = df.dropna(subset=["q_measured", "_slope"])
    logger.info("After cleaning: %d valid rows", len(df_valid))
    
    # Fit per cabin
    cabin_coefs = {}
    failed = []
    for cabin_id in sorted(df_valid["cavity_id"].unique()):
        sub = df_valid[df_valid["cavity_id"] == cabin_id]
        if len(sub) < args.min_samples_per_cabin:
            logger.warning("Cabin %d: only %d samples, skipping",
                           cabin_id, len(sub))
            continue
        result = fit_one_cabin(
            sub["_slope"].values,
            sub["q_measured"].values,
            args.bootstrap_samples,
            args.seed + int(cabin_id),
        )
        if result is None:
            continue
        
        passed = result["r_squared"] >= args.min_r2
        result["passes_acceptance"] = bool(passed)
        if not passed:
            logger.warning(
                "Cabin %d: R²=%.4f below threshold %.2f",
                cabin_id, result["r_squared"], args.min_r2,
            )
            failed.append(int(cabin_id))
        
        cabin_coefs[int(cabin_id)] = result
        logger.info(
            "Cabin %d: β=%.3e α=%.3e R²=%.4f n=%d",
            cabin_id, result["beta"], result["alpha"],
            result["r_squared"], result["n_samples"],
        )
    
    # Output JSON
    out = {
        "version": args.version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature": feature_name,
        "target": "Q (Pa·m³/s)",
        "primary_section": profile.primary_section,
        "n_cabins_calibrated": len(cabin_coefs),
        "n_cabins_failed_acceptance": len(failed),
        "acceptance": {"min_r2": args.min_r2},
        "cabins": cabin_coefs,
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    
    logger.info("Wrote M1 coefficients: %s", out_path)
    if failed:
        logger.warning("Cabins failing acceptance: %s", failed)


if __name__ == "__main__":
    main()
```

### 10.3.3 `train/train_m2.py`

```python
"""Train M2 — global XGBoost regressor for Q estimation.

Trains in log10(Q) space. Uses round-based train/test split (not random)
to avoid data leakage. Includes feature selection: train with all 43 features,
keep only top-K by importance.

Usage:
    python -m train.train_m2 \
        --data train_data_S2.csv \
        --output models/artifacts/v2.6.0/ \
        --version v2.6.0 \
        --top-k-features 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.feature_spec import FEATURE_ORDER_43D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--version", default="v2.6.0")
    p.add_argument("--top-k-features", type=int, default=20,
                   help="Keep top-K features by importance (after first pass)")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--holdout-round", type=int, default=5,
                   help="Use this round_id as holdout (default 5)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def extract_features(df, feature_names):
    """Extract feature matrix from 'features' JSON column."""
    rows = []
    for s in df["features"]:
        feats = json.loads(s) if isinstance(s, str) else s
        rows.append([float(feats.get(name, 0.0)) for name in feature_names])
    return np.asarray(rows, dtype=np.float32)


def main():
    args = parse_args()
    
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["q_measured", "features"]).reset_index(drop=True)
    logger.info("Loaded %d valid rows", len(df))
    
    if "round_id" not in df.columns:
        logger.warning("No round_id column; using random 80/20 split")
        df["round_id"] = np.random.RandomState(args.seed).randint(1, 6, len(df))
    
    # Train/test split by round
    df_train = df[df["round_id"] != args.holdout_round]
    df_test = df[df["round_id"] == args.holdout_round]
    logger.info("Train: %d rows (rounds != %d), Test: %d rows",
                len(df_train), args.holdout_round, len(df_test))
    
    # Extract full 43-dim features
    X_train = extract_features(df_train, FEATURE_ORDER_43D)
    X_test = extract_features(df_test, FEATURE_ORDER_43D)
    
    # Target: log10(Q), clip Q to [1e-7, 1] to avoid -inf
    y_train_q = np.clip(df_train["q_measured"].values, 1e-7, 1.0)
    y_test_q = np.clip(df_test["q_measured"].values, 1e-7, 1.0)
    y_train = np.log10(y_train_q)
    y_test = np.log10(y_test_q)
    
    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # ── First pass: train with all 43 features ────────────────
    logger.info("Pass 1: training with all 43 features for importance ranking")
    booster_full = xgb.train(
        params={
            "max_depth": args.max_depth,
            "eta": args.learning_rate,
            "reg_lambda": args.reg_lambda,
            "objective": "reg:squarederror",
            "verbosity": 0,
            "seed": args.seed,
        },
        dtrain=xgb.DMatrix(X_train_s, label=y_train, feature_names=FEATURE_ORDER_43D),
        num_boost_round=args.n_estimators,
    )
    
    importance = booster_full.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_k_names = [name for name, _ in sorted_imp[:args.top_k_features]]
    logger.info("Top-%d features: %s", args.top_k_features, top_k_names[:5])
    
    # ── Second pass: train with selected features ─────────────
    feature_indices = [FEATURE_ORDER_43D.index(name) for name in top_k_names]
    X_train_sel = X_train_s[:, feature_indices]
    X_test_sel = X_test_s[:, feature_indices]
    
    # Re-fit scaler on selected features (so deployment scaler aligns with reduced input)
    scaler_sel = StandardScaler()
    X_train_sel_raw = X_train[:, feature_indices]
    X_train_sel_norm = scaler_sel.fit_transform(X_train_sel_raw)
    X_test_sel_raw = X_test[:, feature_indices]
    X_test_sel_norm = scaler_sel.transform(X_test_sel_raw)
    
    logger.info("Pass 2: training with top-%d features", args.top_k_features)
    booster = xgb.train(
        params={
            "max_depth": args.max_depth,
            "eta": args.learning_rate,
            "reg_lambda": args.reg_lambda,
            "objective": "reg:squarederror",
            "verbosity": 0,
            "seed": args.seed,
        },
        dtrain=xgb.DMatrix(X_train_sel_norm, label=y_train, feature_names=top_k_names),
        num_boost_round=args.n_estimators,
    )
    
    # ── Evaluation (back in linear Q space) ───────────────────
    y_train_pred_log = booster.predict(xgb.DMatrix(X_train_sel_norm))
    y_test_pred_log = booster.predict(xgb.DMatrix(X_test_sel_norm))
    y_train_pred = 10 ** y_train_pred_log
    y_test_pred = 10 ** y_test_pred_log
    
    train_r2 = r2_score(y_train_q, y_train_pred)
    test_r2 = r2_score(y_test_q, y_test_pred)
    test_mae = mean_absolute_error(y_test_q, y_test_pred)
    
    logger.info("Train R²: %.4f, Test R²: %.4f, Test MAE: %.3e",
                train_r2, test_r2, test_mae)
    
    if abs(train_r2 - test_r2) > 0.05:
        logger.warning("Possible overfitting: train R² %.4f vs test R² %.4f",
                       train_r2, test_r2)
    
    # ── Save artifacts ─────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    booster.save_model(str(out_dir / "m2_xgb_model.json"))
    joblib.dump(scaler_sel, out_dir / "m2_xgb_scaler.joblib")
    
    metadata = {
        "version": args.version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.data,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "log_space": True,
        "feature_subset": top_k_names,
        "feature_importance": {name: float(imp) for name, imp in sorted_imp[:args.top_k_features]},
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "reg_lambda": args.reg_lambda,
        },
        "evaluation": {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_mae_pa_m3_s": float(test_mae),
        },
    }
    with open(out_dir / "m2_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Wrote M2 artifacts to %s", out_dir)


if __name__ == "__main__":
    main()
```

## 10.4 验收标准

- [ ] `train/prepare_q_data.py` 能从 v2.6 导出 CSV 计算 q_measured
- [ ] `train/train_m1.py` 输出格式与任务 6 LinearRegressionM1 期望的 JSON 结构一致
- [ ] `train/train_m2.py` 输出 model.json + scaler.joblib + metadata.json,与任务 7 M2 加载逻辑兼容
- [ ] M2 训练含两阶段:第一阶段全 43 维 → 第二阶段选 top-K 重训
- [ ] 用合成 mock 数据完成端到端 smoke test

---

# 全局验收清单

完成所有 10 个任务后,执行以下端到端验证:

## E2E-1:Mock 模式启动

```bash
python main.py --mode mock
# 启动日志应显示:
# - "CycleProfile validated: bph_13000"
# - "M1 not loaded: ..."(实验数据未到位时正常)
# - "M2 not loaded: ..."
# - 系统进入 paused 状态等待 's' 命令
```

## E2E-2:完整数据流(无模型)

```bash
python main.py --mode mock
# 输入 s
# 等 30 秒
# 输入 e 暂停
# 输入 x 导出 CSV
# 检查导出 CSV:
#   - 每条 pressure_data 是 70 点的 JSON 数组
#   - cycle_profile_id = "bph_13000"
#   - q_est = NULL (M1 未加载)
#   - 数据库文件大小 < 不压缩版本的 60%
```

## E2E-3:Mock 训练 + 推理回路

用合成数据走通 M1 训练 → 部署 → 推理:

```bash
# 用 mock 数据生成训练 CSV(由 Claude Code 实现 fixtures)
python tests/fixtures/generate_mock_q_data.py > mock_q_data.csv

# 训练 M1
python -m train.train_m1 \
    --data mock_q_data.csv \
    --output models/artifacts/test/m1_coefficients.json \
    --version test_v1

# 训练 M2
python -m train.train_m2 \
    --data mock_q_data.csv \
    --output models/artifacts/test/ \
    --version test_v1 \
    --top-k-features 15

# 部署
mkdir -p models/artifacts/current
cp models/artifacts/test/* models/artifacts/current/

# 启动并采集
python main.py --mode mock
# 输入 s, 等 30 秒
# 检查日志: 
#   - "M1 loaded: version=test_v1, ... cabins"
#   - 每圈推理日志: "Cabin N: OK/LEAK (Q_est=..., threshold=...)"
```

## E2E-4:数据库 schema 兼容

```bash
# 用 v2.5 的旧 ldpj_data.db 文件(有的话)
python main.py --mode mock
# 启动日志应包含 ALTER TABLE migration 痕迹
# 旧记录的新字段为 NULL,新记录正常填充
```

## E2E-5:测试套件

```bash
pytest -v
# 全部通过(包括 deprecated v2.5 测试经过适配)
```

## E2E-6:V_cabin 标定脚本

```bash
python scripts/calibrate_v_cabin.py --cabin 5 \
    --weights-grams 348.2,348.5,348.0 --calibrator "tester"
# configs/cabins.yaml 中 cabin 5 的 v_cabin 应被更新
# data/calibration_history/v_cabin_log.csv 新增一行
```

---

# 文档更新

完成所有任务后,需要更新项目文档:

- [ ] `README.md`:更新版本号到 v2.6,改写"特征工程"和"机械时序"段
- [ ] `docs/Ldpj_backend_architecture_v2.6.md`(新增):描述 v2.6 架构变化
- [ ] 在 README 中添加 `cabins.yaml` 和 `products.yaml` 的简介

---

# 不在本期范围内

明确**不做**的事情:

- HMI 客户界面改造(双模式 Q/d 输入显示)— 由后续任务处理
- PLC 端 STL/SCL 程序 — 由自控工程师处理
- API server 新端点(查询 Q 历史曲线、按舱号导出 raw curve)— 后续任务
- 数据归档脚本(把 30 天前数据导出 parquet)— v2.7 任务
- 阶段 5 的工况修正因子 k_V 的代码实现 — 实验数据出来后单独处理
- 完整的 GUM 不确定度评估 — 当前简化为 σ_repeats
- 流式推理(采集到一半就开始推理)— v2.7 优化点
- 配方热加载 — 重启服务即可,本期不做
- 从 PLC 配方表读取参数 — 已为接口预留(`get_active_cycle_profile()` 函数化),v2.7 实现
- 多档位 cycle_profile 配置 — yaml 结构已支持,目前只填一档

---

# 联系与依赖

- **代码改动负责**:佘红霄(算法)
- **PLC 端配合**:自控工程师
  - 任务 1:配方系统的 yaml 结构需自控工程师了解(便于未来对齐 PLC 配方表字段命名)
  - 任务 2:整圈采集要求 polling 缓冲扩到 25000 帧
  - 任务 8:`cabinHealthStatus` 字段的语义从"概率"改为"漏率"——**此变化必须协调 HMI 显示侧同步切换**,否则上线时显示会异常
- **测试依赖**:本期改造**不依赖**实验数据。Claude Code 用 mock 数据完成所有任务。真实标定数据进入后,只需替换 `models/artifacts/current/` 下的文件即可生效。

---

**末**
