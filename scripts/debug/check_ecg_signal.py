"""
检查 ECG 信号质量
"""
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

ecg_path = 'data/raw/ds005873-download/sub-001/ses-01/ecg/sub-001_ses-01_task-szMonitoring_run-01_ecg.edf'

print(f"检查 ECG 文件: {ecg_path}")
print("="*80)

with pyedflib.EdfReader(ecg_path) as edf:
    print(f"通道数: {edf.signals_in_file}")
    
    for i in range(edf.signals_in_file):
        label = edf.getLabel(i)
        fs = edf.getSampleFrequency(i)
        print(f"  通道 {i}: {label}, 采样率: {fs} Hz")
    
    # 读取第一个通道（应该是 ECG）
    ecg_signal = edf.readSignal(0)
    fs = edf.getSampleFrequency(0)
    
    print(f"\nECG 信号统计:")
    print(f"  长度: {len(ecg_signal)} 样本 ({len(ecg_signal)/fs:.1f} 秒)")
    print(f"  均值: {np.mean(ecg_signal):.6f}")
    print(f"  标准差: {np.std(ecg_signal):.6f}")
    print(f"  最小值: {np.min(ecg_signal):.6f}")
    print(f"  最大值: {np.max(ecg_signal):.6f}")
    print(f"  非零值数量: {np.count_nonzero(ecg_signal)}")
    print(f"  零值占比: {(len(ecg_signal) - np.count_nonzero(ecg_signal)) / len(ecg_signal) * 100:.2f}%")
    
    # 绘制前10秒信号
    duration = 10
    samples = int(duration * fs)
    time = np.arange(samples) / fs
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, ecg_signal[:samples])
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.title(f'ECG 信号（前 {duration} 秒）')
    plt.grid(True)
    plt.savefig('ecg_signal_preview.png', dpi=100, bbox_inches='tight')
    print(f"\n✅ 信号图已保存: ecg_signal_preview.png")
    
    # 检查是否全零
    if np.all(ecg_signal == 0):
        print("\n❌ 警告：ECG 信号全为零！")
    elif np.std(ecg_signal) < 0.0001:
        print(f"\n❌ 警告：ECG 信号几乎无变化（标准差 = {np.std(ecg_signal):.10f}）")
    else:
        print(f"\n✅ ECG 信号有效（标准差 = {np.std(ecg_signal):.6f}）")
