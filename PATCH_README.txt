# Per-Cabin Write-Back + Angle Trigger + Export Fix Patch (v2.4)
# 按舱写回 + 角度触发 + 导出修复 补丁包
#
# 修复内容:
# 1. 按舱写回: 适配 DB_Global v1 (20字节 CabinParam UDT)
# 2. 角度触发: 压力下降触发 → 角度触发 (v2.4 核心修复)
# 3. 导出命令: 按 x 可导出数据到 CSV (原来未接入)
# 4. Cabin[0] 预留, 实际 Cabin[1]~Cabin[25]
# 
# 使用方法:
#   1. 解压到项目根目录，覆盖同名文件
#   2. 重启服务: sudo systemctl restart ldpj_backend
#
# ══════════════════════════════════════════════════════════════
# 变更文件清单 (11 个文件)
# ══════════════════════════════════════════════════════════════
#
# configs/plc.yaml               — CabinParam 20字节, Cabin[0]预留
# configs/runtime.yaml           — v2.4 角度触发, threshold=0.25
# core/cycle_fsm.py              — ★核心★ 角度触发FSM
# core/polling_engine.py         — 20字节帧, Mock读写回环
# integration/result_sender.py   — 按舱写回, 跳过Cabin[0]
# pipeline/processing_loop.py    — write_result加cabin_id
# pipeline/control.py            — 帮助文本增加 x 导出命令
# main.py                        — 注册 x 导出命令 + active_start/end
# tests/conftest.py              — fixture适配
# tests/test_cycle_fsm.py        — 角度触发单元测试
#
# ══════════════════════════════════════════════════════════════
# 修复详情
# ══════════════════════════════════════════════════════════════
#
# [修复1] 角度触发 (cycle_fsm.py)
#   IDLE→COLLECTING: angle从<100°穿越到>=100° (上升沿)
#   COLLECTING→PROCESSING: angle>=276° 且 点数>=20
#   备用结束: max_points(45) / duration(3.6s)
#   超时→FAULT: 8s
#
# [修复2] 按舱写回 (result_sender.py)
#   cabin_base = cabin_id × 20
#   leakTestResult_AI → DB9.DBX (base+12).0
#   cabinHealthStatus → DB9.DBD (base+14)
#
# [修复3] 导出命令 (main.py + control.py)
#   按 x 调用 export_to_csv() → 导出到 export_YYYYMMDD_HHMMSS.csv
#   原因: data_exporter.py 已存在但未注册到 CommandController
