# DB9 CabinParam 结构修正补丁

## 问题根因

对照 `DB_Global [DB9]` PLC 文档, 发现代码中 `CabinParam` 的大小假设有误:

| 项目 | 代码原值 | PLC 实际值 |
|------|---------|-----------|
| cabin_size_bytes | **12** | **18** |
| cabin_count | 25 | **26** (Cabin[0]..Cabin[25]) |
| 字段数 | 4 个 | **7 个** |

遗漏的 3 个字段: `leakTestResult_AI` (Bool), `leakTestResult_PLC` (Bool), `cabinHealthStatus` (Real)

**后果**: 从 Cabin[1] 起, 每个舱室的解析偏移量偏差 6 字节, 所有传感器数据读取错误.

---

## 修改文件清单

### 1. `configs/plc.yaml`
- `cabin_size_bytes`: 12 → **18**
- `cabin_count`: 25 → **26**
- `fields` 列表补充 3 个字段及其偏移量
- `write_back` 从固定偏移改为 **per_cabin** 模式 (写回各舱室的 `leakTestResult_AI` bit)
- 新增 `global_fields` 段 (MainGearPosition, HMI_CabinShield, Motor 等)

### 2. `core/polling_engine.py`
- `CabinFrame` 新增: `leak_result_ai`, `leak_result_plc`, `cabin_health_status`
- `_parse_frame()`: 解析完整 18 字节, 含 Bool bit 提取和 Real 读取
- `MockS7Connection`: 生成 18 字节结构体, 模拟健康度数据

### 3. `core/cycle_fsm.py`
- `CycleData` 新增: `leak_results_plc: List[bool]`, `health_statuses: List[float]`
- `_append()`: 采集时同步记录 PLC 检测结果和健康度

### 4. `integration/result_sender.py`
- 新增 **per_cabin** 写回模式: 按舱室写 `leakTestResult_AI` bit
- `write_result()` 签名变更: 增加 `cabin_id` 参数
- 新增 `write_health_status()` 方法
- 保留 `fixed` 模式向后兼容

### 5. `pipeline/processing_loop.py`
- `write_result()` 调用新增 `cabin_id` 参数
- `db.log_record()` 传入 `leak_results_plc` 和 `health_statuses`
- 新增 `_derive_plc_label()`: 模型缺失时利用 PLC 结果自动标注

### 6. `storage/database_logger.py`
- 建表语句新增 `leak_results_plc TEXT` 和 `health_statuses TEXT`
- `log_record()` 新增两个可选参数 (向后兼容)
- 新增 `_run_migrations()`: 自动为已有数据库添加新列

### 7. 测试文件
- `tests/conftest.py`: fixture 更新
- `tests/test_polling_engine.py`: 适配 18 字节 / 26 舱室
- `tests/test_cycle_fsm.py`: 验证新 CycleData 字段
- `tests/test_database.py`: 验证新列读写

---

## 部署方式

将 `fixes/` 目录下的文件覆盖到项目对应路径:

```bash
cp fixes/configs/plc.yaml           Ldpj_backend/configs/plc.yaml
cp fixes/core/polling_engine.py     Ldpj_backend/core/polling_engine.py
cp fixes/core/cycle_fsm.py          Ldpj_backend/core/cycle_fsm.py
cp fixes/integration/result_sender.py Ldpj_backend/integration/result_sender.py
cp fixes/pipeline/processing_loop.py Ldpj_backend/pipeline/processing_loop.py
cp fixes/storage/database_logger.py Ldpj_backend/storage/database_logger.py
cp fixes/tests/conftest.py          Ldpj_backend/tests/conftest.py
cp fixes/tests/test_polling_engine.py Ldpj_backend/tests/test_polling_engine.py
cp fixes/tests/test_cycle_fsm.py    Ldpj_backend/tests/test_cycle_fsm.py
cp fixes/tests/test_database.py     Ldpj_backend/tests/test_database.py
```

对于已有数据库, `DatabaseLogger` 会在启动时自动执行 ALTER TABLE 添加新列, 无需手动迁移.

---

## write_result 接口变更 (Breaking Change)

`ResultSender.write_result()` 签名从:

```python
write_result(label: int, probability: float)
```

变更为:

```python
write_result(cabin_id: int, label: int, probability: float)
```

如有其他代码直接调用此方法, 需同步更新.
