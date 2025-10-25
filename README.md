# 癫痫发作预测系统 (Epilepsy Seizure Prediction System)

基于心率变异性(HRV)的癫痫发作预测系统，使用机器学习方法对心电信号进行分析和分类。

## 📁 项目结构

```
Watch/
├── src/                        # 核心功能模块
│   ├── __init__.py
│   ├── hrv_features.py        # HRV特征提取（时域+频域）
│   └── fft_transform.py       # FFT变换实现
├── utils/                      # 工具类
│   ├── __init__.py
│   ├── ring_buffer.py         # 环形缓冲区
│   └── micro_tar.py           # 轻量级TAR归档工具
├── scripts/                    # 脚本
│   ├── train_hrv_model.py     # 模型训练脚本
│   └── hrv_server.py          # Flask API服务器
├── visualization/              # 可视化
│   └── epilepsy_event_plotter.py  # 癫痫事件绘图
├── c_extensions/              # C语言扩展（高性能FFT）
│   ├── fft.c
│   ├── fft.h
│   └── README.md
├── data/                      # 数据目录
│   └── rr_dataset.csv        # RR间期数据集
├── models/                    # 模型目录（自动创建）
├── config.py                  # 配置文件
├── requirements.txt           # Python依赖
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python scripts/train_hrv_model.py data/rr_dataset.csv models/model.pkl
```

### 3. 启动API服务器

```bash
python scripts/hrv_server.py
```

服务器将在 `http://localhost:8000` 启动。

### 4. 使用API进行预测

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rr_ms": [800, 820, 810, 830, 825, 815, 805, 810]}'
```

### 5. 可视化癫痫事件

```bash
python visualization/epilepsy_event_plotter.py data/sample_dataset.csv output/
```

## 📊 HRV特征说明

系统提取以下HRV特征：

**时域特征：**
- `meanNN`: RR间期平均值
- `sdnn`: RR间期标准差
- `rmssd`: RR间期差值均方根
- `pnn50`: 相邻RR间期差>50ms的百分比
- `sd1`, `sd2`: Poincaré图参数
- `sd2_sd1`: SD2/SD1比值

**频域特征：**
- `lf`: 低频功率 (0.04-0.15 Hz)
- `hf`: 高频功率 (0.15-0.40 Hz)
- `lf_hf`: LF/HF比值
- `lfn`: 归一化低频功率
- `hfn`: 归一化高频功率

## 🔧 配置说明

在 `config.py` 中可以修改以下参数：

```python
FS_ACCEL = 25               # 采样率(Hz)
WIN_SEC = 60                # 滑窗长度(s)
HOP_SEC = 15                # 滑窗步长(s)
PREICTAL_MINUTES = 15       # 预发作窗口(分钟)
INTERICTAL_GAP_MIN = 50     # 间歇期间隔(分钟)
```

## 📦 模块功能

### src/hrv_features.py
提供HRV特征提取功能：
- `compute_time_domain()`: 时域特征
- `compute_freq_domain()`: 频域特征
- `compute_hrv_features()`: 综合特征提取

### src/fft_transform.py
FFT变换实现：
- `FFTTransformer`: 实数FFT
- `FFTCosqTransformer`: 余弦四分之一波变换

### utils/ring_buffer.py
环形缓冲区，用于流式数据处理

### utils/micro_tar.py
轻量级TAR归档工具，用于数据打包

## 🌐 API接口

### POST /predict
预测HRV分类

**请求体：**
```json
{
  "rr_ms": [800, 820, 810, 830, 825, 815, 805, 810]
}
```

**响应：**
```json
{
  "probs": [0.1, 0.7, 0.2],
  "predicted_class": 1
}
```

### GET /health
健康检查

**响应：**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🎯 标签说明

- **0**: Normal (正常)
- **1**: Pre-Ictal (预发作期)
- **2**: Ictal (发作期)

## 📝 数据格式

RR间期数据集格式 (`data/rr_dataset.csv`)：

```csv
rr_ms,label
"[800,820,810,830,825]",0
"[750,760,755,765,770]",1
```

## 🔬 性能优化

- 默认使用Python版FFT (NumPy)
- 如需极致性能，可编译 `c_extensions/` 中的C版本
- 模型使用RandomForest，可根据需求调整超参数

## 📄 许可证

本项目仅供学习和研究使用。

## 👥 贡献

欢迎提交Issue和Pull Request！

## 📮 联系方式

如有问题，请通过Issue联系。

---

**注意**: 本系统仅用于研究目的，不可用于实际医疗诊断。
