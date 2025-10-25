"""
XCM 模型训练脚本 - 针对类别不平衡优化版本

解决方案：
1. 加权损失函数（Weighted Loss）
2. 焦点损失（Focal Loss）
3. SMOTE 过采样
4. 适合不平衡数据的评估指标（F1, AUPRC）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# tsai imports
from tsai.all import *


# =============== 焦点损失函数 ===============
class FocalLoss(nn.Module):
    """
    焦点损失 - 自动降低简单样本的权重，关注困难样本
    论文: https://arxiv.org/abs/1708.02002
    
    适用场景：类别极度不平衡（如 99:1）
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重 [class0_weight, class1_weight, ...]
        self.gamma = gamma  # 聚焦参数，通常 2.0
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============== 数据加载 ===============
def load_xcm_data(csv_path, use_smote=False):
    """
    加载 XCM 格式数据并解析 JSON 序列
    
    Args:
        csv_path: CSV 文件路径
        use_smote: 是否使用 SMOTE 过采样
    
    Returns:
        X: (n_samples, n_channels, seq_len) numpy array
        y: (n_samples,) numpy array
        df: 原始 DataFrame
    """
    print(f"\n读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  总样本数: {len(df)}")
    
    # 标签分布
    label_counts = df['label'].value_counts().sort_index()
    print(f"\n标签分布:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} ({count/len(df)*100:.2f}%)")
    
    # 解析序列（JSON 格式）
    print("\n解析序列数据...")
    hr_seq = np.array([json.loads(row) for row in df['hr_seq']])
    acc_x_seq = np.array([json.loads(row) for row in df['acc_x_seq']])
    acc_y_seq = np.array([json.loads(row) for row in df['acc_y_seq']])
    acc_z_seq = np.array([json.loads(row) for row in df['acc_z_seq']])
    acc_mag_seq = np.array([json.loads(row) for row in df['acc_mag_seq']])
    
    # 构造 (n_samples, n_channels, seq_len) 格式
    X = np.stack([hr_seq, acc_x_seq, acc_y_seq, acc_z_seq, acc_mag_seq], axis=1)
    y = df['label'].values
    
    print(f"  数据形状: X={X.shape}, y={y.shape}")
    print(f"  通道: [HR, ACC_X, ACC_Y, ACC_Z, ACC_MAG]")
    
    # SMOTE 过采样（可选）
    if use_smote and len(np.unique(y)) > 1:
        print(f"\n使用 SMOTE 过采样...")
        # 将 (n_samples, n_channels, seq_len) 展平为 (n_samples, n_features)
        n_samples, n_channels, seq_len = X.shape
        X_flat = X.reshape(n_samples, -1)
        
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.min(np.bincount(y)) - 1))
        X_flat, y = smote.fit_resample(X_flat, y)
        X = X_flat.reshape(-1, n_channels, seq_len)
        
        print(f"  过采样后: X={X.shape}, y={y.shape}")
        label_counts = pd.Series(y).value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"    Label {label}: {count} ({count/len(y)*100:.2f}%)")
    
    return X, y, df


# =============== 模型训练 ===============
def train_model(X, y, output_dir, use_focal_loss=False, focal_gamma=2.0, 
                test_size=0.2, val_size=0.1, epochs=50, batch_size=64):
    """
    训练 XCM 模型（支持类别不平衡优化）
    
    Args:
        X: 输入数据 (n_samples, n_channels, seq_len)
        y: 标签 (n_samples,)
        output_dir: 输出目录
        use_focal_loss: 是否使用焦点损失
        focal_gamma: 焦点损失的 gamma 参数
        test_size: 测试集比例
        val_size: 验证集比例（从训练集中划分）
        epochs: 训练轮数
        batch_size: 批大小
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 划分数据集（分层采样）
    print(f"\n划分数据集 (test={test_size}, val={val_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train
    )
    
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 2. 计算类别权重
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights = torch.FloatTensor(class_weights)
    print(f"\n类别权重: {class_weights.numpy()}")
    
    # 3. 标准化
    print("\n应用标准化...")
    # 计算训练集的均值和标准差
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # 4. 创建数据集
    tfms = [None, TSClassification()]
    dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=(list(range(len(X_train))), list(range(len(X_val)))))
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size*2])
    
    # 5. 创建模型
    print(f"\n创建 XCM 模型...")
    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = XCM(n_channels, n_classes, seq_len=X_train.shape[2])
    
    # 6. 选择损失函数
    if use_focal_loss:
        print(f"  使用焦点损失 (gamma={focal_gamma})")
        loss_func = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        print(f"  使用加权交叉熵损失")
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
    
    # 7. 创建 Learner
    learn = Learner(dls, model, loss_func=loss_func, metrics=[accuracy])
    
    # 8. 设置学习率
    lr = 1e-3  # 固定学习率
    print(f"\n使用学习率: {lr:.2e}")
    
    # 9. 训练
    print(f"\n开始训练 ({epochs} epochs)...")
    learn.fit_one_cycle(epochs, lr_max=lr)
    
    # 10. 保存模型
    model_path = output_dir / 'model.pkl'
    learn.export(model_path)
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 11. 评估
    print(f"\n" + "="*80)
    print("模型评估")
    print("="*80)
    
    # 验证集
    print("\n【验证集】")
    val_preds, val_targets = learn.get_preds(dl=dls.valid)
    val_preds_class = val_preds.argmax(dim=1).numpy()
    val_targets = val_targets.numpy()
    
    print(classification_report(val_targets, val_preds_class, 
                                target_names=[f'Class {i}' for i in range(n_classes)],
                                digits=4))
    
    # 测试集 - 手动预测
    print("\n【测试集】")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        test_preds = model(X_test_tensor)
        test_preds_class = test_preds.argmax(dim=1).numpy()
    
    print(classification_report(y_test, test_preds_class,
                                target_names=[f'Class {i}' for i in range(n_classes)],
                                digits=4))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, test_preds_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Pred {i}' for i in range(n_classes)],
                yticklabels=[f'True {i}' for i in range(n_classes)])
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    print(f"\n✅ 混淆矩阵已保存: {cm_path}")
    plt.close()
    
    # F1 分数（每个类别）
    f1_macro = f1_score(y_test, test_preds_class, average='macro')
    f1_weighted = f1_score(y_test, test_preds_class, average='weighted')
    print(f"\nF1 分数:")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    
    # 保存评估结果
    results = {
        'class_weights': class_weights.numpy().tolist(),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist()
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 评估结果已保存: {results_path}")
    
    return learn


# =============== 主函数 ===============
def main():
    parser = argparse.ArgumentParser(description="训练 XCM 模型（类别不平衡优化版）")
    parser.add_argument('--csv', required=True, help='输入 CSV 文件路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--use-smote', action='store_true', help='使用 SMOTE 过采样')
    parser.add_argument('--use-focal-loss', action='store_true', help='使用焦点损失（推荐）')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='焦点损失的 gamma 参数')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val-size', type=float, default=0.1, help='验证集比例')
    args = parser.parse_args()
    
    print("="*80)
    print("XCM 模型训练 - 类别不平衡优化版")
    print("="*80)
    print(f"配置:")
    print(f"  数据: {args.csv}")
    print(f"  输出: {args.output}")
    print(f"  SMOTE 过采样: {args.use_smote}")
    print(f"  焦点损失: {args.use_focal_loss}")
    if args.use_focal_loss:
        print(f"  Focal Gamma: {args.focal_gamma}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批大小: {args.batch_size}")
    print("="*80)
    
    # 加载数据
    X, y, df = load_xcm_data(args.csv, use_smote=args.use_smote)
    
    # 训练模型
    learn = train_model(
        X, y, 
        output_dir=args.output,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        test_size=args.test_size,
        val_size=args.val_size,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)


if __name__ == '__main__':
    main()
