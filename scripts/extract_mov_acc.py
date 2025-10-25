"""
从 MOV EDF 文件中提取加速度（ACC）数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mne
import json
import argparse


def extract_acc_features(acc_x, acc_y, acc_z, window_size=125, step_size=25):
    """
    从加速度信号提取特征（滑窗方式）
    
    Args:
        acc_x, acc_y, acc_z: 三轴加速度数据
        window_size: 窗口大小（采样点数），默认 125（对应 25Hz 采样下的 5 秒）
        step_size: 步长，默认 25（1 秒）
    
    Returns:
        features: 每个窗口的特征字典列表
    """
    n_samples = len(acc_x)
    features = []
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        
        # 提取窗口数据
        x_win = acc_x[start:end]
        y_win = acc_y[start:end]
        z_win = acc_z[start:end]
        
        # 计算特征
        # 1. 均值
        mean_x, mean_y, mean_z = np.mean(x_win), np.mean(y_win), np.mean(z_win)
        
        # 2. 标准差
        std_x, std_y, std_z = np.std(x_win), np.std(y_win), np.std(z_win)
        
        # 3. 合成加速度（向量模）
        magnitude = np.sqrt(x_win**2 + y_win**2 + z_win**2)
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        
        # 4. 信号变化率（近似导数）
        diff_x, diff_y, diff_z = np.diff(x_win), np.diff(y_win), np.diff(z_win)
        mean_diff = np.mean(np.abs(np.concatenate([diff_x, diff_y, diff_z])))
        
        features.append({
            'window_start': start,
            'window_end': end,
            'time_start': start / 25.0,  # 假设 25 Hz
            'time_end': end / 25.0,
            'mean_x': float(mean_x),
            'mean_y': float(mean_y),
            'mean_z': float(mean_z),
            'std_x': float(std_x),
            'std_y': float(std_y),
            'std_z': float(std_z),
            'mean_magnitude': float(mean_mag),
            'std_magnitude': float(std_mag),
            'mean_diff': float(mean_diff)
        })
    
    return features


def extract_mov_acc_from_edf(edf_path):
    """
    从 MOV EDF 文件提取加速度数据
    
    Args:
        edf_path: MOV EDF 文件路径
    
    Returns:
        dict: {
            'acc_x': x 轴加速度,
            'acc_y': y 轴加速度,
            'acc_z': z 轴加速度,
            'fs': 采样率,
            'duration': 时长,
            'features': 窗口特征列表
        }
    """
    print(f"读取 MOV EDF: {edf_path}")
    
    # 读取 EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # 查找加速度通道（可能名称：ACC_X, ACC_Y, ACC_Z 或 AccX, AccY, AccZ）
    channel_names = raw.ch_names
    print(f"  可用通道: {channel_names}")
    
    acc_channels = {}
    for axis in ['X', 'Y', 'Z']:
        candidates = [ch for ch in channel_names if axis.upper() in ch.upper() and 'ACC' in ch.upper()]
        if candidates:
            acc_channels[axis] = candidates[0]
    
    if len(acc_channels) < 3:
        raise ValueError(f"未找到完整的 ACC_X/Y/Z 通道，可用: {channel_names}")
    
    print(f"  使用加速度通道: {acc_channels}")
    
    # 提取数据
    acc_x = raw.get_data(picks=[acc_channels['X']])[0]
    acc_y = raw.get_data(picks=[acc_channels['Y']])[0]
    acc_z = raw.get_data(picks=[acc_channels['Z']])[0]
    
    fs = raw.info['sfreq']
    duration = len(acc_x) / fs
    
    print(f"  采样率: {fs} Hz, 时长: {duration:.1f} 秒")
    print(f"  数据形状: {acc_x.shape}")
    
    # 提取特征
    print(f"  提取加速度特征（滑窗）...")
    features = extract_acc_features(acc_x, acc_y, acc_z)
    print(f"  提取了 {len(features)} 个窗口特征")
    
    return {
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'fs': fs,
        'duration': duration,
        'features': features
    }


def save_acc_data(acc_data, output_path, save_raw=False):
    """
    保存加速度数据
    
    Args:
        acc_data: 加速度数据字典
        output_path: 输出 JSON 路径
        save_raw: 是否保存原始信号（可能很大）
    """
    save_data = {
        'fs': float(acc_data['fs']),
        'duration': float(acc_data['duration']),
        'n_samples': len(acc_data['acc_x']),
        'n_windows': len(acc_data['features']),
        'features': acc_data['features']
    }
    
    # 可选：保存原始信号（会很大，通常不建议）
    if save_raw:
        save_data['acc_x'] = acc_data['acc_x'].tolist()
        save_data['acc_y'] = acc_data['acc_y'].tolist()
        save_data['acc_z'] = acc_data['acc_z'].tolist()
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"✅ ACC 数据已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="从 MOV EDF 提取加速度数据")
    parser.add_argument('--edf', required=True, help='输入 MOV EDF 文件路径')
    parser.add_argument('--output', required=True, help='输出 JSON 文件路径')
    parser.add_argument('--save-raw', action='store_true', help='是否保存原始信号（文件会很大）')
    args = parser.parse_args()
    
    try:
        # 提取加速度数据
        acc_data = extract_mov_acc_from_edf(args.edf)
        
        # 保存结果
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_acc_data(acc_data, args.output, save_raw=args.save_raw)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
