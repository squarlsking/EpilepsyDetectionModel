# train_hrv_model.py
"""
训练HRV分类模型
用法: python train_hrv_model.py <数据集路径> <输出模型路径>
"""
import json
import joblib
import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hrv_features import compute_hrv_features

def rr_list_from_str(s):
    """从字符串解析RR间期列表"""
    try:
        return np.array(json.loads(s))
    except:
        return np.array([float(x) for x in s.replace(',', ' ').split() if x.strip()!=''])

def extract_features(rr_ms):
    """提取HRV特征向量"""
    feats = compute_hrv_features(rr_ms)
    keys = ["meanNN","sdnn","rmssd","pnn50","sd1","sd2","sd2_sd1",
            "lf","hf","lf_hf","lfn","hfn"]
    return [feats.get(k,0.0) for k in keys]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python train_hrv_model.py <训练数据CSV> <输出模型路径>")
        print("示例: python train_hrv_model.py data/rr_dataset.csv models/model.pkl")
        exit(1)
    
    csv_path = sys.argv[1]
    out_model = sys.argv[2]

    print(f"[INFO] 加载数据集: {csv_path}")
    df = pd.read_csv(csv_path)
    
    X, y = [], []
    for idx, row in df.iterrows():
        rr = rr_list_from_str(row["rr_ms"])
        feats = extract_features(rr)
        X.append(feats)
        y.append(int(row["label"]))

    X, y = np.array(X), np.array(y)
    print(f"[INFO] 数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"[INFO] 标签分布: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] 训练随机森林模型...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("[INFO] 模型评估:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 如果是二分类，计算ROC AUC
    if len(np.unique(y)) == 2:
        y_prob = clf.predict_proba(X_test)[:,1]
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # 保存模型
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    model_pack = {
        "model": clf,
        "feature_keys": ["meanNN","sdnn","rmssd","pnn50","sd1","sd2","sd2_sd1",
                        "lf","hf","lf_hf","lfn","hfn"]
    }
    joblib.dump(model_pack, out_model)
    print(f"[SUCCESS] 模型已保存到: {out_model}")
