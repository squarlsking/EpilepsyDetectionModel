import neurokit2 as nk
import pyedflib
import matplotlib.pyplot as plt
import numpy as np

ecg_path = 'data/raw/ds005873-download/sub-001/ses-01/ecg/sub-001_ses-01_task-szMonitoring_run-01_ecg.edf'

# === 读取 ECG ===
with pyedflib.EdfReader(ecg_path) as edf:
    ecg_signal = edf.readSignal(0)
    fs = edf.getSampleFrequency(0)

print(f"采样率: {fs} Hz")

# 取前 30 秒测试
duration = 30
ecg_segment = ecg_signal[:int(duration * fs)]

# === NeuroKit2 自动处理 ===
signals, info = nk.ecg_process(ecg_segment, sampling_rate=fs)

# 获取R峰索引和清洁后的ECG信号
r_peaks = info["ECG_R_Peaks"]
ecg_cleaned = signals["ECG_Clean"]

# 输出峰值信息
print(f"\nR峰数量: {len(r_peaks)}")
print(f"平均心率: {np.mean(signals['ECG_Rate']):.1f} bpm")

# RR间期统计
rr_intervals = np.diff(r_peaks) / fs * 1000
rr_mean = np.mean(rr_intervals)
rr_std = np.std(rr_intervals)
print(f"RR 平均: {rr_mean:.1f} ± {rr_std:.1f} ms")

# 检测异常RR间期
short_rr_mask = rr_intervals < rr_mean - 2 * rr_std
long_rr_mask = rr_intervals > rr_mean + 2 * rr_std
print(f"\nRR 间期异常检测:")
print(f"  异常短的 RR (< {rr_mean - 2*rr_std:.0f} ms): {np.sum(short_rr_mask)} 个")
print(f"  异常长的 RR (> {rr_mean + 2*rr_std:.0f} ms): {np.sum(long_rr_mask)} 个")
if np.sum(short_rr_mask) > 0:
    print(f"  ⚠️  可能存在误检的小波峰！最短 RR: {np.min(rr_intervals):.0f} ms")
else:
    print(f"  ✅ 未检测到误检的小波峰")

# === 自定义可视化（清晰的红点图） ===
plt.figure(figsize=(16, 10))

# 时间轴
time = np.arange(len(ecg_segment)) / fs

# 1. 原始信号 + R峰
plt.subplot(3, 1, 1)
plt.plot(time, ecg_segment, 'b-', alpha=0.6, linewidth=0.8, label='Original ECG')
plt.plot(r_peaks / fs, ecg_segment[r_peaks], 'ro', markersize=8, label=f'R-peaks (n={len(r_peaks)})')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original ECG + R-peak Detection (NeuroKit2)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([0, duration])

# 2. 清洁后信号 + R峰（放大显示QRS波群细节）
plt.subplot(3, 1, 2)
plt.plot(time, ecg_cleaned, 'g-', linewidth=1, label='Cleaned ECG')
plt.plot(r_peaks / fs, ecg_cleaned[r_peaks], 'ro', markersize=8, label=f'R-peaks (n={len(r_peaks)})')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Cleaned ECG + R-peak Detection (Check for False Peaks)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([0, duration])

# 3. RR间期时间序列（检查稳定性）
plt.subplot(3, 1, 3)
if len(r_peaks) > 1:
    rr_times = r_peaks[1:] / fs
    plt.plot(rr_times, rr_intervals, 'b.-', markersize=5, linewidth=1, label='RR intervals')
    plt.axhline(rr_mean, color='g', linestyle='--', alpha=0.7, linewidth=2, label=f'Mean: {rr_mean:.0f}ms')
    plt.axhline(rr_mean + 2*rr_std, color='r', linestyle='--', alpha=0.5, label=f'±2σ')
    plt.axhline(rr_mean - 2*rr_std, color='r', linestyle='--', alpha=0.5)
    
    # 标记异常点
    if np.sum(short_rr_mask) > 0:
        plt.plot(rr_times[short_rr_mask], rr_intervals[short_rr_mask], 'rx', 
                markersize=12, markeredgewidth=3, label='Abnormally short (false peak?)')
    if np.sum(long_rr_mask) > 0:
        plt.plot(rr_times[long_rr_mask], rr_intervals[long_rr_mask], 'mo', 
                markersize=10, markeredgewidth=2, label='Abnormally long')
    
    plt.xlabel('Time (s)')
    plt.ylabel('RR Interval (ms)')
    plt.title('RR Interval Time Series (Stability Check)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, duration])
    plt.ylim([max(0, rr_mean - 3*rr_std), rr_mean + 3*rr_std])

plt.tight_layout()
plt.savefig("ecg_r_peak_neurokit2.png", dpi=100, bbox_inches='tight')
print(f"\n✅ 图表已保存: ecg_r_peak_neurokit2.png")

# === 对比建议 ===
print(f"\n" + "="*80)
print("NeuroKit2 优势:")
print("  ✅ 专业的ECG信号处理库，算法经过验证")
print("  ✅ 自动信号清洁（去噪、基线漂移校正）")
print("  ✅ 多种R峰检测算法可选（默认使用 neurokit 算法）")
print("  ✅ 自动计算心率变异性（HRV）指标")
print("="*80)
