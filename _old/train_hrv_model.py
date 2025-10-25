# train_hrv_model.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from features_hrv import compute_hrv_features

def rr_list_from_str(s):
    try:
        return np.array(json.loads(s))
    except:
        return np.array([float(x) for x in s.replace(',', ' ').split() if x.strip()!=''])

def extract_features(rr_ms):
    feats = compute_hrv_features(rr_ms)
    keys = ["meanNN","sdnn","rmssd","pnn50","sd1","sd2","sd2_sd1",
            "lf","hf","lf_hf","lfn","hfn"]
    return [feats.get(k,0.0) for k in keys]

if __name__ == "__main__":
    import sys
    if len(sys.argv)<3:
        print("Usage: python train_hrv_model.py train.csv model.pkl")
        exit(1)
    csv_path, out_model = sys.argv[1], sys.argv[2]

    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        rr = rr_list_from_str(row["rr_ms"])
        feats = extract_features(rr)
        X.append(feats)
        y.append(int(row["label"]))

    X, y = np.array(X), np.array(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    clf = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    print(classification_report(y_test,y_pred))
    print("ROC AUC:", roc_auc_score(y_test,y_prob))

    joblib.dump({
        "model": clf,
        "feature_keys": ["meanNN","sdnn","rmssd","pnn50","sd1","sd2","sd2_sd1","lf","hf","lf_hf","lfn","hfn"]
    }, out_model)
    print("Model saved to", out_model)
