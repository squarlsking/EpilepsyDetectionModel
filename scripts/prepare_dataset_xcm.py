"""
主数据处理流程（XCM 格式版本）：整合 ECG/RR、MOV/ACC、标签，生成 tsai XCM 训练数据集

改进：
- 30秒窗口，15秒步长（50%重叠）
- 4Hz 统一采样频率（减轻模型压力）
- 输出 HR 和 ACC 序列（每个窗口 120 个时间步）
- 丰富的 ACC 特征（SMA, Energy, Jerk, Correlations, 频域）
- 符合 tsai XCM 多变量时间序列格式

用法：
  python scripts/prepare_dataset_xcm.py --data-dir data/raw/ds005873-download --output data/processed/seizure_dataset_xcm.csv
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 信号处理库
from scipy import signal
from scipy.interpolate import interp1d

# 导入自定义模块
from extract_ecg_rr import extract_ecg_rr_from_edf
from extract_mov_acc import extract_mov_acc_from_edf
from parse_annotations import parse_tsv_annotations


# ============ 配置参数 ============
FS_TARGET = 4           # 目标采样频率 (Hz)
WINDOW_SEC = 30         # 窗口大小（秒）
STEP_SEC = 15           # 默认步长（秒，50% 重叠）
SEQ_LEN = WINDOW_SEC * FS_TARGET  # 序列长度 = 120

# 自适应步长策略（针对类别不平衡）
STEP_SEC_INTERICTAL = 30   # 间歇期步长（减少冗余）
STEP_SEC_PREICTAL = 10     # 预发作期步长（增加样本）
STEP_SEC_ICTAL = 5         # 发作期步长（密集采样）

PREICTAL_MIN = 15       # 预发作窗口（分钟）
INTERICTAL_GAP_MIN = 50 # 间歇期最小间隔（分钟）

MIN_RR_COUNT = 15       # 窗口内最少 RR 数量
RR_QUALITY_THRESHOLD = 0.7  # RR 质量阈值


def find_subject_files(data_dir):
    """
    扫描 BIDS 数据集，找到所有受试者的 ECG/MOV EDF 和 TSV 标签文件
    
    Returns:
        list: [{subject, session, run, ecg_edf, mov_edf, tsv}, ...]
    """
    data_path = Path(data_dir)
    subjects = []
    
    # 遍历所有受试者目录（sub-xxx）
    for sub_dir in sorted(data_path.glob('sub-*')):
        subject_id = sub_dir.name
        
        # 直接使用 ses-01（数据集中每个受试者只有一个会话）
        session_id = 'ses-01'
        ses_dir = sub_dir / session_id
        
        if not ses_dir.exists():
            continue
        
        # 查找 ECG 目录（包含 ECG EDF）
        ecg_dir = ses_dir / 'ecg'
        ecg_edfs = list(ecg_dir.glob('*_ecg.edf')) if ecg_dir.exists() else []
        
        # 查找 EEG 目录（包含标签 TSV）
        eeg_dir = ses_dir / 'eeg'
        tsv_files = list(eeg_dir.glob('*_events.tsv')) if eeg_dir.exists() else []
        
        # 查找 MOV 目录（包含 ACC 数据）
        mov_dir = ses_dir / 'mov'
        mov_edfs = list(mov_dir.glob('*_mov.edf')) if mov_dir.exists() else []
        
        # 配对文件（按 run 编号）
        for ecg_edf in ecg_edfs:
            # 提取 run 编号（如 run-09）
            run_id = None
            parts = ecg_edf.stem.split('_')
            for part in parts:
                if part.startswith('run-'):
                    run_id = part
                    break
            
            # 查找对应的 MOV 和 TSV
            mov_edf = None
            tsv_file = None
            
            if run_id:
                mov_candidates = [f for f in mov_edfs if run_id in f.name]
                mov_edf = mov_candidates[0] if mov_candidates else None
                
                tsv_candidates = [f for f in tsv_files if run_id in f.name]
                tsv_file = tsv_candidates[0] if tsv_candidates else None
            
            # 如果有 ECG 和 MOV 就记录
            if ecg_edf and mov_edf:
                subjects.append({
                    'subject': subject_id,
                    'session': session_id,
                    'run': run_id,
                    'ecg_edf': str(ecg_edf),
                    'mov_edf': str(mov_edf),
                    'tsv': str(tsv_file) if tsv_file else None
                })
    
    return subjects


def compute_acc_features(acc_x, acc_y, acc_z, fs=4):
    """
    计算丰富的加速度特征
    
    Args:
        acc_x, acc_y, acc_z: 三轴加速度序列（已重采样到 fs）
        fs: 采样频率
    
    Returns:
        dict: 特征字典
    """
    features = {}
    
    # 合成加速度
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # SMA (Signal Magnitude Area)
    features['acc_sma'] = np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z))
    
    # Energy (RMS)
    features['acc_energy'] = np.sqrt(np.mean(acc_x**2 + acc_y**2 + acc_z**2))
    
    # Jerk (加加速度)
    jerk_mag = np.diff(acc_mag) * fs  # d(mag)/dt
    features['acc_jerk_mean'] = np.mean(np.abs(jerk_mag))
    features['acc_jerk_std'] = np.std(jerk_mag)
    
    # 轴相关性
    features['acc_corr_xy'] = np.corrcoef(acc_x, acc_y)[0, 1] if len(acc_x) > 1 else 0
    features['acc_corr_xz'] = np.corrcoef(acc_x, acc_z)[0, 1] if len(acc_x) > 1 else 0
    features['acc_corr_yz'] = np.corrcoef(acc_y, acc_z)[0, 1] if len(acc_y) > 1 else 0
    
    # 峰值计数（阈值设为均值 + 1.5倍标准差）
    threshold = np.mean(acc_mag) + 1.5 * np.std(acc_mag)
    peaks, _ = signal.find_peaks(acc_mag, height=threshold)
    features['acc_peak_count'] = len(peaks)
    
    # 过零率
    zero_crossings = np.sum(np.diff(np.sign(acc_mag - np.mean(acc_mag))) != 0)
    features['acc_zero_cross'] = zero_crossings
    
    # 频域能量（0.1-3 Hz，捕捉震颤）
    if len(acc_mag) >= 8:  # 至少需要 8 个点做 FFT
        freqs, psd = signal.periodogram(acc_mag, fs=fs)
        low_freq_mask = (freqs >= 0.1) & (freqs <= 3.0)
        features['acc_spectral_energy'] = np.sum(psd[low_freq_mask]) / np.sum(psd) if np.sum(psd) > 0 else 0
    else:
        features['acc_spectral_energy'] = 0
    
    return features


def resample_to_grid(rr_times, rr_ms, acc_times, acc_x, acc_y, acc_z, 
                     start_time, end_time, fs_target=4):
    """
    将 RR 和 ACC 数据重采样到统一的时间网格
    
    Args:
        rr_times: RR 间期时间戳（秒）
        rr_ms: RR 间期值（毫秒）
        acc_times: ACC 采样时间戳（秒）
        acc_x, acc_y, acc_z: 三轴加速度
        start_time, end_time: 窗口时间范围
        fs_target: 目标采样频率（Hz）
    
    Returns:
        dict: {
            'hr_seq': [...],      # 心率序列
            'acc_x_seq': [...],
            'acc_y_seq': [...],
            'acc_z_seq': [...],
            'acc_mag_seq': [...],
            'rr_quality': float   # RR 信号质量
        }
    """
    # 创建统一时间网格
    grid = np.arange(start_time, end_time, 1/fs_target)
    n_points = len(grid)
    
    # 1. RR → HR 插值
    if len(rr_times) >= 2 and len(rr_ms) >= 2:
        # 计算瞬时心率（bpm）
        inst_hr = 60000.0 / np.array(rr_ms)  # 60000 ms/min ÷ rr_ms
        
        # 过滤异常心率（30-200 bpm）
        valid_mask = (inst_hr >= 30) & (inst_hr <= 200)
        valid_times = np.array(rr_times)[valid_mask]
        valid_hr = inst_hr[valid_mask]
        
        if len(valid_times) >= 2:
            # 线性插值
            hr_seq = np.interp(grid, valid_times, valid_hr)
            rr_quality = len(valid_times) / len(rr_times)  # 质量 = 有效率
        else:
            hr_seq = np.full(n_points, 70.0)  # 默认心率
            rr_quality = 0.0
    else:
        hr_seq = np.full(n_points, 70.0)
        rr_quality = 0.0
    
    # 2. ACC → 重采样
    if len(acc_times) >= 2:
        # 线性插值
        interp_x = interp1d(acc_times, acc_x, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(acc_times, acc_y, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
        interp_z = interp1d(acc_times, acc_z, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
        
        acc_x_seq = interp_x(grid)
        acc_y_seq = interp_y(grid)
        acc_z_seq = interp_z(grid)
    else:
        acc_x_seq = np.zeros(n_points)
        acc_y_seq = np.zeros(n_points)
        acc_z_seq = np.full(n_points, 9.81)  # 默认重力加速度
    
    # 计算合成加速度
    acc_mag_seq = np.sqrt(acc_x_seq**2 + acc_y_seq**2 + acc_z_seq**2)
    
    return {
        'hr_seq': hr_seq.tolist(),
        'acc_x_seq': acc_x_seq.tolist(),
        'acc_y_seq': acc_y_seq.tolist(),
        'acc_z_seq': acc_z_seq.tolist(),
        'acc_mag_seq': acc_mag_seq.tolist(),
        'rr_quality': rr_quality
    }


def label_time_segments(time_array, seizure_events, preictal_min=15, interictal_gap_min=50):
    """
    为时间序列打标签
    
    Args:
        time_array: 时间戳数组（秒）
        seizure_events: 癫痫发作事件列表 [(onset, duration), ...]
        preictal_min: 预发作窗口（分钟）
        interictal_gap_min: 间歇期最小间隔（分钟）
    
    Returns:
        labels: 标签数组 (0=间歇期, 1=预发作期, 2=发作期, -1=丢弃)
    """
    labels = np.full(len(time_array), 0, dtype=int)  # 默认间歇期
    
    for onset, duration in seizure_events:
        # 发作期（使用真实的 duration）
        ictal_mask = (time_array >= onset) & (time_array < onset + duration)
        labels[ictal_mask] = 2
        
        # 预发作期
        preictal_start = onset - preictal_min * 60
        preictal_mask = (time_array >= preictal_start) & (time_array < onset)
        labels[preictal_mask] = 1
        
        # 标记太接近发作的间歇期为 -1（丢弃）
        too_close_mask = (np.abs(time_array - onset) < interictal_gap_min * 60) & (labels == 0)
        labels[too_close_mask] = -1
    
    return labels


def process_subject(subject_info, output_interim_dir):
    """
    处理单个受试者的数据（XCM 格式）
    
    Returns:
        list: 训练样本列表
    """
    subject = subject_info['subject']
    session = subject_info['session']
    run = subject_info['run']
    
    print(f"\n{'='*60}")
    print(f"处理: {subject}/{session}/{run}")
    print(f"{'='*60}")
    
    samples = []
    
    try:
        # 1. 提取 RR 间期
        rr_data = None
        if subject_info['ecg_edf']:
            print(f"\n[1/3] 提取 ECG/RR...")
            rr_data = extract_ecg_rr_from_edf(subject_info['ecg_edf'])
        
        # 2. 提取 ACC 原始数据
        acc_data = None
        if subject_info['mov_edf']:
            print(f"\n[2/3] 提取 MOV/ACC...")
            acc_data = extract_mov_acc_from_edf(subject_info['mov_edf'])
        
        # 3. 解析标签
        seizure_events = []
        if subject_info['tsv']:
            print(f"\n[3/3] 解析标签...")
            annotations = parse_tsv_annotations(subject_info['tsv'])
            
            # 提取癫痫发作事件（parse_tsv_annotations 已经筛选了 sz 开头的事件）
            for ann in annotations:
                if 'onset' in ann and 'duration' in ann:
                    seizure_events.append((ann['onset'], ann['duration']))
            
            print(f"  检测到 {len(seizure_events)} 个癫痫发作事件")
        
        # 4. 滑窗处理（30秒窗口，15秒步长）
        if rr_data and acc_data:
            print(f"\n生成训练样本（30s窗口，4Hz采样）...")
            
            # 确定时间范围
            max_time = min(rr_data['rr_times'][-1] if len(rr_data['rr_times']) > 0 else 0,
                          acc_data['features'][-1]['time_end'] if len(acc_data['features']) > 0 else 0)
            
            if max_time < WINDOW_SEC:
                print(f"  ⚠️  数据太短 ({max_time:.1f}s)，跳过")
                return samples
            
            # 准备 ACC 原始数据（从 features 中提取，假设是按时间顺序的窗口）
            # 注意：extract_mov_acc_from_edf 返回的是5秒窗口的统计特征，这里需要原始信号
            # 我们需要修改 extract_mov_acc.py 来返回原始信号，或者这里直接读 EDF
            
            # 临时方案：从 EDF 直接读取 ACC 原始数据
            import pyedflib
            with pyedflib.EdfReader(subject_info['mov_edf']) as edf:
                # 查找 ACC 通道
                signal_labels = [edf.getLabel(i).strip().upper() for i in range(edf.signals_in_file)]
                
                acc_x_idx = next((i for i, label in enumerate(signal_labels) if 'ACC' in label and 'X' in label), None)
                acc_y_idx = next((i for i, label in enumerate(signal_labels) if 'ACC' in label and 'Y' in label), None)
                acc_z_idx = next((i for i, label in enumerate(signal_labels) if 'ACC' in label and 'Z' in label), None)
                
                if acc_x_idx is None or acc_y_idx is None or acc_z_idx is None:
                    print(f"  ⚠️  未找到 ACC 通道")
                    return samples
                
                # 读取原始信号
                acc_x_raw = edf.readSignal(acc_x_idx)
                acc_y_raw = edf.readSignal(acc_y_idx)
                acc_z_raw = edf.readSignal(acc_z_idx)
                acc_fs = edf.getSampleFrequency(acc_x_idx)
                
                # 创建时间戳
                acc_times = np.arange(len(acc_x_raw)) / acc_fs
            
            # 自适应滑窗（根据标签动态调整步长）
            window_count = 0
            window_start = 0.0
            
            while window_start < max_time - WINDOW_SEC:
                window_end = window_start + WINDOW_SEC
                window_center = (window_start + window_end) / 2
                
                # 获取窗口内的 RR 数据
                rr_mask = (np.array(rr_data['rr_times']) >= window_start) & \
                         (np.array(rr_data['rr_times']) < window_end)
                window_rr_times = np.array(rr_data['rr_times'])[rr_mask]
                window_rr_ms = np.array(rr_data['rr_ms'])[rr_mask]
                
                # 检查 RR 质量
                if len(window_rr_ms) < MIN_RR_COUNT:
                    # 使用默认步长跳过
                    window_start += STEP_SEC
                    continue
                
                # 获取窗口内的 ACC 数据
                acc_mask = (acc_times >= window_start) & (acc_times < window_end)
                window_acc_x = acc_x_raw[acc_mask]
                window_acc_y = acc_y_raw[acc_mask]
                window_acc_z = acc_z_raw[acc_mask]
                window_acc_times = acc_times[acc_mask]
                
                if len(window_acc_x) < 10:
                    continue
                
                # 重采样到 4Hz 网格
                resampled = resample_to_grid(
                    window_rr_times, window_rr_ms,
                    window_acc_times, window_acc_x, window_acc_y, window_acc_z,
                    window_start, window_end, fs_target=FS_TARGET
                )
                
                # 检查 RR 质量
                if resampled['rr_quality'] < RR_QUALITY_THRESHOLD:
                    continue
                
                # 判断标签
                if len(seizure_events) > 0:
                    label = label_time_segments(np.array([window_center]), seizure_events, 
                                               PREICTAL_MIN, INTERICTAL_GAP_MIN)[0]
                else:
                    label = 0  # 没有发作，全部标为间歇期
                
                # 跳过标记为 -1 的样本
                if label == -1:
                    # 使用默认步长跳过
                    window_start += STEP_SEC
                    continue
                
                # 根据标签选择下一步的步长（自适应采样）
                if label == 0:
                    next_step = STEP_SEC_INTERICTAL
                elif label == 1:
                    next_step = STEP_SEC_PREICTAL
                else:  # label == 2
                    next_step = STEP_SEC_ICTAL
                
                # 计算 ACC 特征
                acc_features = compute_acc_features(
                    np.array(resampled['acc_x_seq']),
                    np.array(resampled['acc_y_seq']),
                    np.array(resampled['acc_z_seq']),
                    fs=FS_TARGET
                )
                
                # 构建样本
                sample = {
                    'subject': subject,
                    'session': session,
                    'run': run,
                    'time_start': window_start,
                    'time_end': window_end,
                    'label': int(label),
                    
                    # 序列数据（JSON 字符串）
                    'hr_seq': json.dumps(resampled['hr_seq']),
                    'acc_x_seq': json.dumps(resampled['acc_x_seq']),
                    'acc_y_seq': json.dumps(resampled['acc_y_seq']),
                    'acc_z_seq': json.dumps(resampled['acc_z_seq']),
                    'acc_mag_seq': json.dumps(resampled['acc_mag_seq']),
                    
                    # 质量和统计特征
                    'n_rr_original': len(window_rr_ms),
                    'rr_quality': resampled['rr_quality'],
                    
                    # ACC 特征
                    **acc_features
                }
                
                samples.append(sample)
                window_count += 1
                
                # 更新窗口起始位置（使用自适应步长）
                window_start += next_step
            
            print(f"✅ 生成了 {window_count} 个样本")
            print(f"  采样策略: 间歇期={STEP_SEC_INTERICTAL}s, 预发作={STEP_SEC_PREICTAL}s, 发作期={STEP_SEC_ICTAL}s")
            
            # 打印标签分布
            if samples:
                labels_count = pd.Series([s['label'] for s in samples]).value_counts().sort_index()
                print(f"  标签分布: {labels_count.to_dict()}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="处理 SeizeIT2 数据集，生成 tsai XCM 格式训练数据")
    parser.add_argument('--data-dir', required=True, help='BIDS 数据集根目录')
    parser.add_argument('--output', required=True, help='输出 CSV 文件路径')
    parser.add_argument('--interim-dir', default='data/interim', help='中间结果目录')
    parser.add_argument('--max-subjects', type=int, default=None, help='限制处理的受试者数量（测试用）')
    args = parser.parse_args()
    
    print("="*80)
    print("SeizeIT2 数据集处理 - XCM 格式版本")
    print("="*80)
    print(f"配置:")
    print(f"  窗口大小: {WINDOW_SEC}s")
    print(f"  步长: {STEP_SEC}s")
    print(f"  采样频率: {FS_TARGET} Hz")
    print(f"  序列长度: {SEQ_LEN} (= {WINDOW_SEC}s × {FS_TARGET}Hz)")
    print(f"  预发作窗口: {PREICTAL_MIN} 分钟")
    print(f"  间歇期间隔: {INTERICTAL_GAP_MIN} 分钟")
    print("="*80)
    
    # 1. 扫描数据集
    print("\n[阶段 1] 扫描数据集...")
    subjects = find_subject_files(args.data_dir)
    
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
        print(f"  限制处理前 {args.max_subjects} 个受试者")
    
    print(f"  找到 {len(subjects)} 个数据文件组合")
    
    # 2. 处理每个受试者
    print("\n[阶段 2] 处理数据...")
    all_samples = []
    
    for subject_info in tqdm(subjects, desc="处理进度"):
        samples = process_subject(subject_info, args.interim_dir)
        all_samples.extend(samples)
    
    # 3. 保存结果
    print(f"\n[阶段 3] 保存数据集...")
    df = pd.DataFrame(all_samples)
    
    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ 数据处理完成！")
    print(f"{'='*80}")
    print(f"输出文件: {args.output}")
    print(f"总样本数: {len(df)}")
    print(f"序列长度: {SEQ_LEN} (每个样本)")
    
    if len(df) > 0:
        print(f"\n标签分布:")
        print(df['label'].value_counts().sort_index())
        print(f"\nRR 质量统计:")
        print(f"  平均: {df['rr_quality'].mean():.3f}")
        print(f"  最小: {df['rr_quality'].min():.3f}")
        print(f"  最大: {df['rr_quality'].max():.3f}")
        print(f"\n数据集大小: {len(df)} 行 × {len(df.columns)} 列")
        print(f"磁盘占用: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"\n⚠️  警告：没有生成任何样本！")
        print(f"可能原因:")
        print(f"  1. ECG 信号质量太差，检测不到 R 峰")
        print(f"  2. RR 质量低于阈值 ({RR_QUALITY_THRESHOLD})")
        print(f"  3. 数据时长不足 ({WINDOW_SEC}秒)")
        print(f"\n建议:")
        print(f"  - 检查 ECG 信号质量")
        print(f"  - 降低 RR_QUALITY_THRESHOLD（当前 {RR_QUALITY_THRESHOLD}）")
        print(f"  - 降低 MIN_RR_COUNT（当前 {MIN_RR_COUNT}）")
    
    print("="*80)


if __name__ == '__main__':
    main()
