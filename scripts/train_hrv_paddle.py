
import os, sys, json, argparse, math, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils import class_weight
from features_hrv import compute_hrv_features  # 假设和本文件在同一目录下
import paddle
import paddle.nn as nn
import paddle.optimizer as optim

# 工具函数
def rr_str_to_array(s):
    # 将字符串（JSON 或逗号分隔）转为 numpy 数组
    try:
        arr = np.array(json.loads(s))
    except Exception:
        arr = np.array([float(x) for x in s.replace(',', ' ').split() if x.strip()!=''])
    return arr

def compute_tabular_features_from_row(rr_ms):
    # 从单个 RR 序列提取 HRV 特征
    featsd = compute_hrv_features(rr_ms)
    # 确保特征顺序固定，便于后续训练
    keys = ["meanNN","sdnn","rmssd","pnn50","sd1","sd2","sd2_sd1","lf","hf","lf_hf","lfn","hfn"]
    return np.array([featsd.get(k, 0.0) for k in keys], dtype=np.float32), keys

def random_oversample(X, y, random_state=42):
    # 简单过采样：将少数类样本复制，直到与多数类样本数相同
    np.random.seed(random_state)
    unique, counts = np.unique(y, return_counts=True)
    max_cnt = counts.max()
    Xs, ys = [], []
    for cls in unique:
        idx = np.where(y==cls)[0]
        n_repeat = int(np.ceil(max_cnt / len(idx)))
        rep = np.tile(idx, n_repeat)[:max_cnt]
        Xs.append(X[rep])
        ys.append(y[rep])
    X_new = np.vstack(Xs)
    y_new = np.hstack(ys)
    p = np.random.permutation(len(y_new))
    return X_new[p], y_new[p]

def random_undersample(X, y, random_state=42):
    # 简单欠采样：随机丢弃多数类样本，直到与少数类相同
    np.random.seed(random_state)
    unique, counts = np.unique(y, return_counts=True)
    min_cnt = counts.min()
    Xs, ys = [], []
    for cls in unique:
        idx = np.where(y==cls)[0]
        sel = np.random.choice(idx, min_cnt, replace=False)
        Xs.append(X[sel])
        ys.append(y[sel])
    X_new = np.vstack(Xs)
    y_new = np.hstack(ys)
    p = np.random.permutation(len(y_new))
    return X_new[p], y_new[p]

#  Paddle 模型定义
class TabularMLP(nn.Layer):
    # 基于 HRV 特征的多层感知机（MLP）
    def __init__(self, input_dim, hidden=[64,32], dropout=0.2):
        super(TabularMLP, self).__init__()
        layers = []
        in_d = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, 2))  # 二分类输出
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class SeqLSTM(nn.Layer):
    # 基于 RR 时间序列的 LSTM 模型
    def __init__(self, input_dim=1, hidden_size=64, num_layers=1, bidirectional=False, dropout=0.2):
        super(SeqLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
                            direction='bidirectional' if bidirectional else 'forward', dropout=dropout)
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(nn.Linear(fc_in, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):
        # 输入 x: [batch, seq_len, feat_dim] -> LSTM 期望输入 [seq_len, batch, feat_dim]
        x = paddle.transpose(x, [1,0,2])
        out, (h,c) = self.lstm(x)
        last = out[-1]  # 取最后一个时间步的输出
        logits = self.fc(last)
        return logits

#  训练与评估
def compute_class_weights_from_y(y):
    # 根据标签分布计算类别权重，用于处理类别不平衡
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    weight_map = {int(c): float(w) for c,w in zip(classes, weights)}
    return np.array([weight_map.get(0,1.0), weight_map.get(1,1.0)], dtype=np.float32)

def evaluate_preds(y_true, y_pred, interictal_hours=1.0):
    # 计算指标：准确率、灵敏度、特异度、每小时误报率
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    fp_per_h = fp / max(1e-9, interictal_hours)
    return {"Acc":acc, "Sens":sens, "Spec":spec, "FP/h":fp_per_h, "cm":cm}

def train_and_evaluate_tabular(X, y, keys, args):
    # TabularMLP 训练 + 交叉验证
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    fold = 0
    results = []
    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # 数据平衡策略
        if args.balance == "oversample":
            X_train, y_train = random_oversample(X_train, y_train, args.seed)
        elif args.balance == "undersample":
            X_train, y_train = random_undersample(X_train, y_train, args.seed)
        # 类别权重
        weights = compute_class_weights_from_y(y_train) if args.class_weights else np.array([1.0,1.0], dtype=np.float32)
        paddle.set_device(args.device)
        # 构建模型
        model = TabularMLP(input_dim=X.shape[1], hidden=args.hidden)
        model.train()
        criterion = nn.CrossEntropyLoss(weight=paddle.to_tensor(weights))
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=args.lr)
        # 构建数据加载器
        train_tensor_x = paddle.to_tensor(X_train.astype('float32'))
        train_tensor_y = paddle.to_tensor(y_train.astype('int64'))
        val_tensor_x = paddle.to_tensor(X_val.astype('float32'))
        val_tensor_y = paddle.to_tensor(y_val.astype('int64'))
        train_ds = paddle.io.TensorDataset([train_tensor_x, train_tensor_y])
        val_ds = paddle.io.TensorDataset([val_tensor_x, val_tensor_y])
        train_loader = paddle.io.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = paddle.io.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        best_val_acc = 0.0
        best_state = None
        # 训练循环
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
            # 验证
            model.eval()
            all_preds = []
            all_trues = []
            with paddle.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    preds = paddle.argmax(logits, axis=1).numpy()
                    all_preds.extend(preds.tolist())
                    all_trues.extend(yb.numpy().tolist())
            metrics = evaluate_preds(np.array(all_trues), np.array(all_preds), interictal_hours=args.interictal_hours)
            val_acc = metrics["Acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()
        results.append({"fold":fold, "val_metrics":metrics, "best_acc":best_val_acc})
        if args.save_fold:
            path = f"{args.out}_fold{fold}.pdparams"
            paddle.save(best_state, path)
    return results

def train_and_evaluate_sequence(df, args):
    # SeqLSTM 训练 + 交叉验证
    seqs = []
    labels = []
    for _, row in df.iterrows():
        rr = rr_str_to_array(row["rr_ms"])
        seq = rr.astype(np.float32) / 1000.0  # 毫秒转秒
        if len(seq) < args.seq_len:
            pad = np.zeros(args.seq_len - len(seq), dtype=np.float32)
            seq = np.concatenate([seq, pad])
        else:
            seq = seq[:args.seq_len]
        seqs.append(seq.reshape(args.seq_len,1))
        labels.append(int(row["label"]))
    X = np.array(seqs)
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    fold=0; results=[]
    for train_idx, val_idx in skf.split(X,y):
        fold+=1
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # 数据平衡
        if args.balance == "oversample":
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_train_flat, y_train = random_oversample(X_train_flat, y_train, args.seed)
            X_train = X_train_flat.reshape(len(X_train_flat), args.seq_len, 1)
        elif args.balance == "undersample":
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_train_flat, y_train = random_undersample(X_train_flat, y_train, args.seed)
            X_train = X_train_flat.reshape(len(X_train_flat), args.seq_len, 1)
        # 类别权重
        weights = compute_class_weights_from_y(y_train) if args.class_weights else np.array([1.0,1.0], dtype=np.float32)
        paddle.set_device(args.device)
        model = SeqLSTM(input_dim=1, hidden_size=args.hidden_size, num_layers=args.num_layers,
                        bidirectional=args.bidirectional, dropout=args.dropout)
        model.train()
        criterion = nn.CrossEntropyLoss(weight=paddle.to_tensor(weights))
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=args.lr)
        # 数据加载
        train_tensor_x = paddle.to_tensor(X_train.astype('float32'))
        train_tensor_y = paddle.to_tensor(y_train.astype('int64'))
        val_tensor_x = paddle.to_tensor(X_val.astype('float32'))
        val_tensor_y = paddle.to_tensor(y_val.astype('int64'))
        train_ds = paddle.io.TensorDataset([train_tensor_x, train_tensor_y])
        val_ds = paddle.io.TensorDataset([val_tensor_x, val_tensor_y])
        train_loader = paddle.io.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = paddle.io.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        best_val_acc = 0.0; best_state=None
        # 训练循环
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
            # 验证
            model.eval()
            all_preds = []; all_trues=[]
            with paddle.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    preds = paddle.argmax(logits, axis=1).numpy()
                    all_preds.extend(preds.tolist())
                    all_trues.extend(yb.numpy().tolist())
            metrics = evaluate_preds(np.array(all_trues), np.array(all_preds), interictal_hours=args.interictal_hours)
            val_acc = metrics["Acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()
        results.append({"fold":fold, "val_metrics":metrics, "best_acc":best_val_acc})
        if args.save_fold:
            path = f"{args.out}_fold{fold}.pdparams"
            paddle.save(best_state, path)
    return results

#  命令行接口
def parse_args():
    p = argparse.ArgumentParser(description="使用 PaddlePaddle 训练 HRV 模型（特征输入或序列输入）。")
    p.add_argument("--csv", type=str, required=True, help="输入 CSV 文件路径")
    p.add_argument("--out", type=str, required=True, help="输出模型文件前缀")
    p.add_argument("--mode", type=str, choices=["tabular","sequence"], default="tabular", help="选择建模方式")
    p.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    p.add_argument("--epochs", type=int, default=40, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=64, help="批大小")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--class_weights", action="store_true", help="是否使用类别权重")
    p.add_argument("--balance", choices=["none","oversample","undersample"], default="none", help="类别平衡策略")
    p.add_argument("--save_fold", action="store_true", help="是否保存每折的模型参数")
    p.add_argument("--device", type=str, default="cpu", help="计算设备: cpu 或 gpu")
    p.add_argument("--hidden", nargs="+", type=int, default=[64,32], help="MLP 隐藏层大小")
    p.add_argument("--interictal_hours", type=float, default=1.0, help="用于计算 FP/h")
    p.add_argument("--seq_len", type=int, default=180, help="序列模型的输入长度（RR 数量）")
    p.add_argument("--hidden_size", type=int, default=64, help="LSTM 隐藏层大小")
    p.add_argument("--num_layers", type=int, default=1, help="LSTM 层数")
    p.add_argument("--bidirectional", action="store_true", help="是否使用双向 LSTM")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout 比例")
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    if not os.path.exists(args.csv):
        print("未找到 CSV 文件:", args.csv); sys.exit(1)
    df = pd.read_csv(args.csv)
    if args.mode == "tabular":
        X=[]; y=[]
        for _, row in df.iterrows():
            rr = rr_str_to_array(row["rr_ms"])
            feats, keys = compute_tabular_features_from_row(rr)
            X.append(feats)
            y.append(int(row["label"]))
        X = np.vstack(X).astype(np.float32); y = np.array(y, dtype=int)
        print("数据维度:", X.shape, "标签分布:", np.unique(y, return_counts=True))
        results = train_and_evaluate_tabular(X, y, keys, args)
    else:
        results = train_and_evaluate_sequence(df, args)
    print("交叉验证结果汇总:")
    for r in results:
        print(r)
    print("完成。如果使用 --save_fold 参数，将保存每折模型。")

if __name__ == '__main__':
    main()
