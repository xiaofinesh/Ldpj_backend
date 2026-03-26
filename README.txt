# Ldpj_backend 训练脚本补丁包
# 生成日期: 2026-03-20

## 使用方法

将 train/ 目录直接覆盖到项目根目录:

    unzip train_patch_20260320.zip -d /path/to/Ldpj_backend/

或手动复制:

    cp train/train_model.py        /path/to/Ldpj_backend/train/train_model.py
    cp train/train_cv_optimized.py /path/to/Ldpj_backend/train/train_cv_optimized.py

## 改动说明

### train/train_model.py (主训练脚本)
- 交叉验证: 5折 → 10折 StratifiedKFold
- 标签来源: 新增 --label-source=valve (默认), 从 leak_valve_status 列派生标签
- 特征来源: 新增 --use-precomputed, 直接读取 features JSON 列
- 阈值优化: 新增 precision-recall curve 自动搜索最优阈值
- 最终模型: 全量数据训练部署模型 (不再只用80%训练集)
- 验收检查: 自动输出 F1/Recall(LEAK)/Precision(OK) 是否达标
- 过滤器: 新增 --min-feature-max 按特征最大值过滤
- 完全向后兼容: --label-source=column --flip-labels 保留旧逻辑

### train/train_cv_optimized.py (CV优化训练脚本)
- 标签来源: 同上, 新增 --label-source / --use-precomputed
- 修复 bug: leak_valve_statuses → leak_valve_status (列名修正)
- 新增 evaluation_report.txt 和验收检查输出
- metadata.json 增加 label_source / leak_recall / ok_precision 字段

## 新系统数据训练命令

    python -m train.train_model \
        --data export_cleaned.csv \
        --output models/artifacts/v1.0 \
        --version v1.0 \
        --use-precomputed

## 旧系统数据训练命令 (向后兼容)

    python -m train.train_model \
        --data old_data.csv \
        --output models/artifacts/v1.0 \
        --label-source column \
        --label-column prediction \
        --flip-labels \
        --min-pressure -5

## patches/ 目录
包含 unified diff 格式的补丁文件, 如需 git apply 可使用:

    cd Ldpj_backend
    git apply patches/train_model.patch
    git apply patches/train_cv_optimized.patch
