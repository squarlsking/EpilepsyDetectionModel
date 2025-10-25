"""
使用 NeuroKit2 从 EDF 文件提取 ECG 信号并计算 RR 间期
优化版：分段处理 + 进度条 + 质量过滤
"""
import neurokit2 as nk
import numpy as np
import mne
import json
import os
from tqdm import tqdm

def extract_ecg_rr_from_edf(edf_path, ecg_channel_name='ECG', segment_duration=300):
    """
    快速提取 RR 间期，仅使用 nk.ecg_peaks（分段处理）
    
    Args:
        edf_path (str): EDF 文件路径
        ecg_channel_name (str): ECG 通道名称关键字（默认 'ECG'）
        segment_duration (int): 分段长度（秒），默认 300s（5分钟）
    
    Returns:
        dict: 包含 'rr_ms', 'rr_times', 'fs', 'duration', 'n_peaks', 'n_valid_rr'
    """
    # === 1. 读取 EDF 文件 ===
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    ch_names = raw.ch_names
    ecg_channels = [ch for ch in ch_names if ecg_channel_name.upper() in ch.upper()]
    if not ecg_channels:
        raise ValueError(f"未找到 ECG 通道，可用通道: {ch_names}")
    
    ecg_channel = ecg_channels[0]
    fs = raw.info['sfreq']
    ecg = raw.get_data(picks=[ecg_channel])[0]
    duration = len(ecg) / fs
    
    print(f"  通道: {ecg_channel} | 采样率: {fs} Hz | 时长: {duration/3600:.2f} 小时")

    # === 2. 分段检测 R 峰（避免内存爆炸） ===
    seg_len = int(segment_duration * fs)
    n_segments = int(np.ceil(len(ecg) / seg_len))
    all_peaks = []

    print(f"  分为 {n_segments} 段（每段 {segment_duration}s），开始检测 R 峰...")
    for i in tqdm(range(n_segments), desc="  R峰检测", unit="段"):
        start = i * seg_len
        end = min((i + 1) * seg_len, len(ecg))
        seg = ecg[start:end]
        
        try:
            _, info = nk.ecg_peaks(seg, sampling_rate=fs)
            peaks = info["ECG_R_Peaks"] + start  # 转换为全局索引
            all_peaks.extend(peaks)
        except Exception as e:
            print(f"\n  ⚠️  第 {i+1}/{n_segments} 段检测失败: {e}")
            continue

    # === 3. 计算 RR 间期 ===
    all_peaks = np.array(all_peaks)
    if len(all_peaks) < 2:
        print(f"  ⚠️  R 峰数量过少（{len(all_peaks)}），无法计算 RR 间期")
        return {
            'rr_ms': np.array([]),
            'rr_times': np.array([]),
            'fs': float(fs),
            'duration': float(duration),
            'n_peaks': int(len(all_peaks)),
            'n_valid_rr': 0
        }
    
    rr_ms = np.diff(all_peaks) / fs * 1000  # 毫秒
    rr_times = all_peaks[1:] / fs           # 秒

    # === 4. 过滤异常 RR 间期（300-2000 ms，对应 30-200 bpm） ===
    valid_mask = (rr_ms >= 300) & (rr_ms <= 2000)
    rr_ms_filtered = rr_ms[valid_mask]
    rr_times_filtered = rr_times[valid_mask]

    n_removed = len(rr_ms) - len(rr_ms_filtered)
    print(f"  检测到 {len(all_peaks)} 个 R 峰")
    print(f"  有效 RR 间期: {len(rr_ms_filtered)} 个 (过滤掉 {n_removed} 个异常值)")
    
    if len(rr_ms_filtered) > 0:
        print(f"  RR 统计: 均值={np.mean(rr_ms_filtered):.1f} ms, 标准差={np.std(rr_ms_filtered):.1f} ms")

    return {
        'rr_ms': rr_ms_filtered,
        'rr_times': rr_times_filtered,
        'fs': float(fs),
        'duration': float(duration),
        'n_peaks': int(len(all_peaks)),
        'n_valid_rr': int(len(rr_ms_filtered))
    }

def save_rr_data(rr_data, output_path):
    """保存 RR 数据到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换 numpy 数组为列表
    save_data = {
        'rr_ms': rr_data['rr_ms'].tolist() if hasattr(rr_data['rr_ms'], 'tolist') else rr_data['rr_ms'],
        'rr_times': rr_data['rr_times'].tolist() if hasattr(rr_data['rr_times'], 'tolist') else rr_data['rr_times'],
        'fs': float(rr_data['fs']),
        'duration': float(rr_data['duration']),
        'n_peaks': int(rr_data['n_peaks']),
        'n_valid_rr': int(rr_data['n_valid_rr'])
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"✅ RR 数据已保存: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 NeuroKit2 从 EDF 提取 ECG 并计算 RR 间期")
    parser.add_argument('--edf', required=True, help='输入 EDF 文件路径')
    parser.add_argument('--output', required=True, help='输出 JSON 文件路径')
    parser.add_argument('--channel', default='ECG', help='ECG 通道名称关键字（默认: ECG）')
    parser.add_argument('--segment', type=int, default=300, help='分段长度（秒，默认: 300）')
    args = parser.parse_args()

    try:
        print(f"\n读取 EDF: {args.edf}")
        rr_data = extract_ecg_rr_from_edf(args.edf, args.channel, args.segment)
        save_rr_data(rr_data, args.output)
        print("\n✅ 处理完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
