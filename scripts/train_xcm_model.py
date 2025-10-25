"""
使用 tsai XCM 模型训练癫痫发作预测

用法：
  python scripts/train_xcm_model.py --csv data/processed/seizure_dataset_xcm.csv --output models/xcm_seizure
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# tsai 框架
from tsai.all import *


def load_xcm_data(csv_path):
    """
    从 CSV 加载数据并转换为 tsai XCM 格式
    
    Returns:
        X: (n_samples, n_channels, seq_len)
        y: (n_samples,)
        df: 原始 DataFrame
    """
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"  原始数据: {len(df)} 样本")
    
    # 解析 JSON 序列
    print("  解析序列数据...")
    df['hr_seq'] = df['hr_seq'].apply(json.loads)
    df['acc_x_seq'] = df['acc_x_seq'].apply(json.loads)
    df['acc_y_seq'] = df['acc_y_seq'].apply(json.loads)
    df['acc_z_seq'] = df['acc_z_seq'].apply(json.loads)
    df['acc_mag_seq'] = df['acc_mag_seq'].apply(json.loads)
    
    # 构建 X (多变量时间序列)
    n_samples = len(df)
    n_channels = 5  # HR, ACC_X, ACC_Y, ACC_Z, ACC_MAG
    seq_len = len(df.iloc[0]['hr_seq'])
    
    print(f"  构建张量: ({n_samples}, {n_channels}, {seq_len})")
    X = np.zeros((n_samples, n_channels, seq_len), dtype=np.float32)
    
    for i, row in df.iterrows():
        X[i, 0, :] = row['hr_seq']        # 通道0: 心率
        X[i, 1, :] = row['acc_x_seq']     # 通道1: X轴加速度
        X[i, 2, :] = row['acc_y_seq']     # 通道2: Y轴加速度
        X[i, 3, :] = row['acc_z_seq']     # 通道3: Z轴加速度
        X[i, 4, :] = row['acc_mag_seq']   # 通道4: 合成加速度
    
    y = df['label'].values
    
    print(f"\n数据集统计:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  标签分布: {pd.Series(y).value_counts().sort_index().to_dict()}")
    
    return X, y, df


def train_xcm_model(X, y, output_dir, epochs=50, batch_size=64, lr=1e-3):
    """
    训练 XCM 模型
    
    Args:
        X: (n_samples, n_channels, seq_len)
        y: (n_samples,)
        output_dir: 模型保存目录
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
    """
    print("\n" + "="*80)
    print("训练 XCM 模型")
    print("="*80)
    
    # 划分训练/验证集（按 subject 分割，避免数据泄露）
    # 简单起见，这里用随机分割
    splits = get_splits(y, valid_size=0.2, stratify=True, random_state=42, shuffle=True)
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(splits[0])} 样本")
    print(f"  验证集: {len(splits[1])} 样本")
    
    # 数据标准化 + 标签编码
    tfms = [None, [Categorize()]]
    batch_tfms = [TSStandardize(by_sample=False, by_var=True)]  # 按通道标准化
    
    # 创建数据集
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    
    # 创建 DataLoader
    dls = TSDataLoaders.from_dsets(
        dsets.train, dsets.valid,
        bs=batch_size,
        batch_tfms=batch_tfms,
        num_workers=0
    )
    
    # 创建 XCM 模型
    print(f"\n创建 XCM 模型...")
    print(f"  输入维度: {dls.vars} 通道 × {dls.len} 时间步")
    print(f"  输出类别: {dls.c} 类")
    
    model = XCM(dls.vars, dls.c, dls.len)
    
    # 创建 Learner
    learn = Learner(
        dls, model,
        metrics=[accuracy, RocAucBinary() if dls.c == 2 else RocAucMulti()],
        cbs=[ShowGraphCallback()]
    )
    
    # 查找最佳学习率（可选）
    # learn.lr_find()
    
    # 训练
    print(f"\n开始训练（{epochs} epochs）...")
    learn.fit_one_cycle(epochs, lr_max=lr)
    
    # 保存模型
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    learn.export(output_path / 'model.pkl')
    print(f"\n✅ 模型已保存: {output_path / 'model.pkl'}")
    
    # 评估
    print(f"\n" + "="*80)
    print("模型评估")
    print("="*80)
    
    # 验证集预测
    probs, targets, preds = learn.get_X_preds(X[splits[1]], y[splits[1]])
    
    # 分类报告
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n分类报告:")
    print(classification_report(targets, preds, target_names=['间歇期', '预发作期', '发作期']))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(targets, preds))
    
    return learn


def main():
    parser = argparse.ArgumentParser(description="训练 tsai XCM 癫痫发作预测模型")
    parser.add_argument('--csv', required=True, help='输入 CSV 文件路径')
    parser.add_argument('--output', required=True, help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--gpu', action='store_true', help='使用 GPU')
    args = parser.parse_args()
    
    # 设置设备
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
    
    # 加载数据
    X, y, df = load_xcm_data(args.csv)
    
    # 训练模型
    learn = train_xcm_model(
        X, y, args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    print(f"\n{'='*80}")
    print("训练完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
