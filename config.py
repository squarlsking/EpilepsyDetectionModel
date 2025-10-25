# config.py
from pathlib import Path

# 采样与滑窗
FS_ACCEL = 25               # 加速度/陀螺仪采样率(Hz)
FS_GYRO  = 25
WIN_SEC  = 60               # 滑窗长度(s)
HOP_SEC  = 15               # 滑窗步长(s)

# HR/HRV
RR_RESAMPLE_HZ = 4          # RR重采样频率(Hz)用于频域HRV
MIN_RR_COUNT_PER_WIN = 30   # 每窗最少RR数量（太少则丢弃）

# 训练
RANDOM_STATE = 42
CV_FOLDS     = 5

# 打标（相对发作起点）
PREICTAL_MINUTES = 15       # 预发作窗口长度（前15分钟）
INTERICTAL_GAP_MIN = 50     # 与发作至少相隔的间隔(分钟)才算间歇期

# 路径
DATA_DIR   = Path("data/")
MODEL_DIR  = Path("models/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
