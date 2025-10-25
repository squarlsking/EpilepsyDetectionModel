import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 关闭无关警告
warnings.filterwarnings("ignore", message=".*set_ticklabels.*")

# 参数配置
DATA_PATH = "../Data/sample_dataset.csv"         # 数据路径
OUTPUT_DIR = "../Sample_Annotations"             # 输出图片路径
SAMPLE_RATE = 25                                 # 假设采样率为 25Hz
WINDOW_SIZE = 125                                # 每个时间窗长度 (点数)
STEP_TIME = WINDOW_SIZE // SAMPLE_RATE           # 每个时间窗对应秒数 (5s)

# 标签对应的颜色
LABEL_COLORS = {
    0: "white",     # Normal
    1: "tab:blue",  # Pre-Ictal
    2: "green"      # Ictal
}

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 绘图函数
def plot_event(event_id, event_data):
    """绘制单个 event 的原始信号与标签分布图"""
    num_points = len(event_data)
    time_axis = np.arange(num_points) / SAMPLE_RATE  # 时间轴 (秒)

    # 每 125 点提取一个标签，表示一个 5s 窗口
    labels = event_data["label"].iloc[::WINDOW_SIZE].reset_index(drop=True)
    label_positions = [i * STEP_TIME for i in range(len(labels))]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # 左轴: 加速度 (rawData)
    ax1.plot(time_axis, event_data["rawData"], color="tab:blue", label="Acceleration (rawData)")
    ax1.set_xlabel("时间 (秒)", fontsize=14)
    ax1.set_ylabel("加速度 (~milli-g)", fontsize=14, color="black")
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12, labelcolor="black")

    # 右轴: 心率 (ppg)
    ax2 = ax1.twinx()
    ax2.plot(time_axis, event_data["ppg"], color="red", label="Heart Rate (ppg)")
    ax2.set_ylabel("心率 (~bp/m)", fontsize=14, color="black")
    ax2.tick_params(axis="y", labelsize=12, labelcolor="black")

    # 顶部: 标签刻度
    ax3 = ax1.twiny()
    ax3.set_xticks(label_positions)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.set_xlabel("真实标签 (Ground Truth)", fontsize=14)
    ax3.tick_params(axis="x", labelsize=10)
    ax3.grid(axis="x", linestyle="--", alpha=0.5)

    # 背景颜色标注 (不同标签区间)
    for i, label in enumerate(labels):
        start = label_positions[i]
        end = label_positions[i + 1] if i + 1 < len(labels) else time_axis[-1]
        ax1.axvspan(start, end, color=LABEL_COLORS.get(label, "white"), alpha=0.2)

    # 图标题
    plt.title(f"Open Seizure Database - Event {event_id}", fontsize=16, pad=20)

    # 图例 (背景 + 曲线)
    custom_legends = [
        mpatches.Patch(color="white", alpha=0.3, label="Normal"),
        mpatches.Patch(color="tab:blue", alpha=0.3, label="Pre-Ictal"),
        mpatches.Patch(color="green", alpha=0.3, label="Ictal"),
        mpatches.Patch(color="tab:blue", alpha=0.7, label="Acceleration"),
        mpatches.Patch(color="red", alpha=0.7, label="Heart Rate"),
    ]
    ax1.legend(handles=custom_legends, loc="lower center", fontsize=12,
               ncol=5, bbox_to_anchor=(0.5, -0.35))

    # 调整布局并保存
    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"event_{event_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ 已保存: {save_path}")

# 主流程

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 按 eventId 分组绘制
    for event_id, event_data in df.groupby("eventId"):
        plot_event(event_id, event_data)
