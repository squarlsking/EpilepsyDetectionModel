# Debug Scripts / 调试脚本

这个目录包含用于开发和调试的辅助脚本。**不用于生产环境**，仅用于：

## 用途

1. **验证数据处理流程** - 快速测试单个步骤是否正确
2. **参数调优** - 测试不同参数对结果的影响
3. **问题排查** - 当出现 bug 时快速定位问题
4. **文档和教学** - 演示如何使用核心功能

## 脚本说明

### `debug_r_peak.py`
- **目的**: 可视化 ECG R 峰检测效果，验证是否误检小波峰
- **使用**: `python scripts/debug/debug_r_peak.py`
- **输出**: `ecg_r_peak_neurokit2.png` - 包含原始/清洁信号 + R峰标记 + RR间期分析
- **何时使用**: 
  - 当怀疑 R 峰检测不准确时
  - 调整检测算法参数时
  - 向他人演示算法效果时

### `check_ecg_channels.py`
- **目的**: 检查 EDF 文件中有哪些通道
- **何时使用**: 不确定数据文件结构时

### `check_ecg_signal.py`
- **目的**: 检查 ECG 信号质量（均值、标准差、是否全零）
- **何时使用**: 怀疑信号质量有问题时



```
scripts/
├── prepare_dataset_xcm.py      # 主要脚本
├── train_xcm_model.py          # 主要脚本
├── extract_ecg_rr.py           # 核心功能
├── parse_annotations.py        # 核心功能
├── debug/                      # 调试脚本目录 ✅
│   ├── README.md
│   ├── debug_r_peak.py
│   ├── check_ecg_channels.py
│   └── check_ecg_signal.py
└── utils/                      # 可复用工具函数
    ├── signal_processing.py    # 例如：通用的滤波函数
    └── visualization.py        # 例如：通用的绘图函数
```

## 版本控制建议

```bash
# .gitignore 中不要忽略 debug 脚本
# 但可以忽略它们的输出
scripts/debug/*.png
scripts/debug/*.csv
```

