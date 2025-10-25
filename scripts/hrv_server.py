# hrv_server.py
"""
Flask服务器 - 用于HRV特征预测
支持本地和AI Studio部署
"""
from flask import Flask, request, jsonify
import os, joblib, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hrv_features import compute_hrv_features

# 路径配置
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/model.pkl")
DATA_PATH = os.environ.get("DATA_PATH", "./data/rr_dataset.csv")

# ========== Step 1: 加载/训练模型 ==========
def load_or_train_model():
    # 1) 已有模型
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading existing model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    # 2) 有数据集 -> 重新训练小模型
    elif os.path.exists(DATA_PATH):
        print("[INFO] No model found, training a new small model...")
        import pandas as pd
        df = pd.read_csv(DATA_PATH)

        X, y = [], []
        for _, row in df.iterrows():
            try:
                rr = np.array(json.loads(row["rr_ms"]), dtype=float)
            except:
                rr = np.array([float(x) for x in row["rr_ms"].replace(',', ' ').split() if x.strip()!=''])
            feats = compute_hrv_features(rr)
            keys = ["meanNN","sdnn","rmssd","pnn50","sd1","sd2","sd2_sd1",
                    "lf","hf","lf_hf","lfn","hfn"]
            vec = [feats.get(k,0.0) for k in keys]
            X.append(vec)
            y.append(int(row["label"]))

        X, y = np.array(X), np.array(y)
        clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X,y)

        model_pack = {"model": clf, "feature_keys": keys}
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model_pack, MODEL_PATH)
        print(f"[INFO] Small model trained and saved to {MODEL_PATH}")
        return model_pack

    # 3) 数据也没有 -> 构建默认模型
    else:
        print("[WARN] No model.pkl or dataset found. Using dummy model.")
        dummy_clf = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
        # 用随机数据拟合，避免 predict 报错
        X_dummy = np.random.rand(10, 12)
        y_dummy = np.random.randint(0, 3, 10)
        dummy_clf.fit(X_dummy, y_dummy)
        keys = ["meanNN","sdnn","rmssd","pnn50","sd1","sd2","sd2_sd1",
                "lf","hf","lf_hf","lfn","hfn"]
        return {"model": dummy_clf, "feature_keys": keys}

model_pack = load_or_train_model()
clf = model_pack["model"]
keys = model_pack["feature_keys"]

# ========== Step 2: Flask API ==========
app = Flask(__name__)

def features_to_vector(rr_ms):
    feats = compute_hrv_features(rr_ms)
    return [feats.get(k,0.0) for k in keys]

@app.route("/predict", methods=["POST"])
def predict():
    j = request.get_json(force=True)
    if "rr_ms" not in j:
        return jsonify({"error":"missing rr_ms"}),400
    rr = np.array(j["rr_ms"],dtype=float)
    vec = np.array(features_to_vector(rr)).reshape(1,-1)
    prob = clf.predict_proba(vec).tolist()[0]  # 多分类概率
    return jsonify({"probs": prob, "predicted_class": int(np.argmax(prob))})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": clf is not None})

if __name__ == "__main__":
    # host 必须是 0.0.0.0，端口可以是 8000 或 8080（AI Studio 默认）
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
