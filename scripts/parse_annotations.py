"""
解析 BIDS 格式的 TSV 标签文件（癫痫发作标注）
支持 SeizeIT2 数据集的 *_events.tsv 格式

TSV 文件示例：
    onset     duration   eventType                    lateralization  localization  vigilance
    5224.00   112.00     sz_foc_a_m_hyperkinetic     left            cen_par       awake
    13745.00  180.00     sz_foc_ia_m_hyperkinetic    left            cen_par       awake
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
import argparse


def parse_tsv_annotations(tsv_path):
    """
    解析 SeizeIT2 格式的 TSV 标签文件（*_events.tsv）
    
    Args:
        tsv_path: TSV 文件路径
    
    Returns:
        list: 癫痫发作事件列表，每个元素为 {onset, offset, duration, event_type, ...}
    """
    print(f"读取标签文件: {tsv_path}")
    
    # 读取 TSV
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"  列名: {df.columns.tolist()}")
    print(f"  总事件数: {len(df)}")
    
    # 检查必需的列
    required_cols = ['onset', 'duration', 'eventType']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"TSV文件缺少必需的列: {col}")
    
    # 筛选癫痫发作事件（eventType 以 'sz' 开头）
    seizure_mask = df['eventType'].str.lower().str.startswith('sz', na=False)
    seizure_df = df[seizure_mask].copy()
    
    print(f"  癫痫发作事件数: {len(seizure_df)}")
    
    # 计算结束时间
    seizure_df['offset'] = seizure_df['onset'] + seizure_df['duration']
    
    # 转换为字典列表
    annotations = seizure_df.to_dict('records')
    
    # 打印统计信息
    if len(seizure_df) > 0:
        event_counts = seizure_df['eventType'].value_counts()
        print(f"\n  发作类型分布:")
        for event_type, count in event_counts.items():
            print(f"    {event_type}: {count}")
        
        print(f"\n  发作时间范围:")
        print(f"    最早发作: {seizure_df['onset'].min():.1f} 秒")
        print(f"    最晚发作: {seizure_df['onset'].max():.1f} 秒")
        print(f"    平均持续: {seizure_df['duration'].mean():.1f} 秒")
    
    return annotations


def merge_annotations_with_data(annotations, rr_times, acc_times, 
                                preictal_minutes=15, interictal_gap_minutes=50):
    """
    将标签与 RR/ACC 数据对齐，生成训练样本
    
    Args:
        annotations: 标签列表（onset, duration）
        rr_times: RR 间期时间戳数组
        acc_times: ACC 窗口时间戳数组
        preictal_minutes: 预发作窗口（分钟）
        interictal_gap_minutes: 间歇期最小间隔（分钟）
    
    Returns:
        list: 标注后的样本列表
    """
    print(f"\n合并标签与数据...")
    print(f"  预发作窗口: {preictal_minutes} 分钟")
    print(f"  间歇期最小间隔: {interictal_gap_minutes} 分钟")
    
    # 提取发作时间（annotations 已经是筛选后的癫痫发作事件）
    seizure_onsets = [ann['onset'] for ann in annotations if 'onset' in ann]
    seizure_durations = [ann.get('duration', 120) for ann in annotations]  # 默认120秒
    
    seizure_events = sorted(zip(seizure_onsets, seizure_durations))
    print(f"  检测到 {len(seizure_events)} 个癫痫发作事件")
    
    # 标注每个时间点
    def get_label(time_sec):
        """
        根据时间判断标签
        0: 间歇期 (Interictal)
        1: 预发作期 (Preictal)
        2: 发作期 (Ictal)
        """
        for onset, duration in seizure_events:
            # 发作期：onset 到 onset + duration
            if onset <= time_sec < onset + duration:
                return 2  # Ictal
            
            # 预发作期：onset 前 preictal_minutes
            preictal_start = onset - preictal_minutes * 60
            if preictal_start <= time_sec < onset:
                return 1  # Preictal
        
        # 间歇期：距离任何发作都足够远
        for onset, duration in seizure_events:
            if abs(time_sec - onset) < interictal_gap_minutes * 60:
                return -1  # 太接近发作，丢弃
        
        return 0  # Interictal
    
    # 标注数据
    labeled_samples = []
    
    # 标注 RR 数据（可按窗口聚合）
    # 标注 ACC 数据
    
    return labeled_samples


def save_annotations(annotations, output_path):
    """保存解析后的标签"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 标签已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="解析 BIDS TSV 标签文件")
    parser.add_argument('--tsv', required=True, help='输入 TSV 文件路径')
    parser.add_argument('--output', required=True, help='输出 JSON 文件路径')
    args = parser.parse_args()
    
    try:
        # 解析标签
        annotations = parse_tsv_annotations(args.tsv)
        
        # 保存结果
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_annotations(annotations, args.output)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
