# 类别不平衡学习策略

癫痫预测任务的核心挑战：**极度类别不平衡**

- Interictal（间歇期）: ~99%
- Preictal（预发作期）: ~0.8%  
- Ictal（发作期）: ~0.07%

## 📊 **问题分析**

### 为什么直接训练效果差？

1. **梯度被主导类淹没** - 模型只学习预测 label=0
2. **评估指标误导** - Accuracy=99% 但完全没学到少数类
3. **决策边界偏移** - 偏向多数类，少数类被忽略

## 🎯 **解决方案对比**

| 方法 | 优点 | 缺点 | 适用场景 | 推荐指数 |
|------|------|------|---------|---------|
| **1. 加权损失** | 简单高效，不改变数据 | 需要调参 | 不平衡比 < 100:1 | ⭐⭐⭐⭐⭐ |
| **2. 焦点损失（Focal Loss）** | 自动关注困难样本 | 计算稍复杂 | 极度不平衡（>100:1） | ⭐⭐⭐⭐⭐ |
| **3. SMOTE 过采样** | 增加少数类样本 | 可能过拟合 | 样本量较小 | ⭐⭐⭐ |
| **4. 欠采样** | 平衡数据集 | 丢失信息 | 数据量充足 | ⭐⭐ |
| **5. 集成方法** | 鲁棒性强 | 训练时间长 | 生产环境 | ⭐⭐⭐⭐ |

---

## ✅ **方法 1: 加权损失函数（推荐首选）**

### 原理
给少数类更高的损失权重，强制模型关注。

### 实现
```python
from sklearn.utils.class_weight import compute_class_weight

# 自动计算权重（基于频率的倒数）
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
# 例如: [0.34, 50.0, 500.0]  # label=2 权重是 label=0 的 1470 倍

# PyTorch 损失函数
loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

### 优点
- ✅ 实现简单，只改变损失函数
- ✅ 不改变数据分布
- ✅ 适用于大多数不平衡场景

### 缺点
- ⚠️ 需要调整权重比例（可能需要手动调整）

---

## 🔥 **方法 2: 焦点损失（Focal Loss）** ⭐ **强烈推荐**

### 原理
自动降低"简单样本"的损失权重，让模型专注于"困难样本"。

$$
FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- $\alpha$: 类别权重
- $\gamma$: 聚焦参数（通常 2.0）
- $(1-p_t)^\gamma$: 调制因子，当样本易分时（$p_t$ 接近 1）权重降低

### 实现
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # [0.34, 50.0, 500.0]
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

### 为什么效果好？
| 样本类型 | 预测概率 | $(1-p_t)^\gamma$ | 损失权重 |
|---------|---------|-----------------|---------|
| 简单正样本 | 0.95 | $(1-0.95)^2 = 0.0025$ | 极低 ⬇️ |
| 困难正样本 | 0.6 | $(1-0.6)^2 = 0.16$ | 中等 |
| 误分类样本 | 0.2 | $(1-0.2)^2 = 0.64$ | 高 ⬆️ |

### 优点
- ✅ 自动关注困难样本
- ✅ 适合极度不平衡（99:1）
- ✅ 经典 YOLO、RetinaNet 都用

### 推荐参数
```python
# 极度不平衡（99:1）
FocalLoss(alpha=class_weights, gamma=2.0)

# 特别极端（999:1）
FocalLoss(alpha=class_weights, gamma=3.0)
```

---

## 📈 **方法 3: SMOTE 过采样**

### 原理
在少数类样本之间插值生成新样本。

### 实现
```python
from imblearn.over_sampling import SMOTE

# 过采样到平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 优点
- ✅ 增加少数类样本数量
- ✅ 避免简单复制（生成新样本）

### 缺点
- ⚠️ 可能过拟合（生成的样本可能不真实）
- ⚠️ 时序数据效果较差（破坏时序依赖）

### 建议
- 仅在样本量 < 1000 时使用
- 结合加权损失一起用

---

## 🎓 **方法 4: 欠采样**

### 原理
随机删除多数类样本使其与少数类平衡。

### 实现
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

### 优点
- ✅ 简单快速

### 缺点
- ❌ 丢失大量信息
- ❌ 数据不足时效果差

### 建议
- **不推荐**用于癫痫预测（本来少数类就少）

---

## 🎯 **评估指标选择（非常重要！）**

### ❌ **不要用 Accuracy！**
```python
# 错误示例：模型预测全是 0，Accuracy 也有 99%
Accuracy = 99%  # 看起来很高，实际啥也没学到
```

### ✅ **应该用的指标**

| 指标 | 说明 | 公式 | 何时使用 |
|------|------|------|---------|
| **Precision** | 预测为正的样本中真正为正的比例 | TP / (TP + FP) | 关注误报（如医疗诊断） |
| **Recall (Sensitivity)** | 真正为正的样本中被正确预测的比例 | TP / (TP + FN) | 关注漏报（如癫痫预测） |
| **F1 Score** | Precision 和 Recall 的调和平均 | $2 \times \frac{P \times R}{P + R}$ | 综合考虑 |
| **AUPRC** | PR 曲线下面积 | - | 极度不平衡首选 ⭐⭐⭐⭐⭐ |
| **AUROC** | ROC 曲线下面积 | - | 类别较平衡时 |

### 推荐配置
```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 1. 分类报告（包含 Precision, Recall, F1）
print(classification_report(y_test, y_pred))

# 2. 每个类别的 F1 分数
f1_macro = f1_score(y_test, y_pred, average='macro')  # 各类别平均
f1_weighted = f1_score(y_test, y_pred, average='weighted')  # 加权平均

# 3. AUPRC（最重要！）
auprc = average_precision_score(y_test, y_pred_proba, average='macro')
```

---

## 🚀 **实战推荐方案**

### **方案 A: 焦点损失（首选）**
```bash
python scripts/train_xcm_imbalanced.py \
  --csv data/processed/seizure_dataset_xcm.csv \
  --output models/xcm_focal \
  --use-focal-loss \
  --focal-gamma 2.0 \
  --epochs 50
```

**适用**: 数据量充足（>10,000 样本）

---

### **方案 B: 焦点损失 + SMOTE**
```bash
python scripts/train_xcm_imbalanced.py \
  --csv data/processed/seizure_dataset_xcm.csv \
  --output models/xcm_focal_smote \
  --use-focal-loss \
  --use-smote \
  --epochs 50
```

**适用**: 数据量较少（<5,000 样本）

---

### **方案 C: 仅加权损失（快速测试）**
```bash
python scripts/train_xcm_imbalanced.py \
  --csv data/processed/seizure_dataset_xcm.csv \
  --output models/xcm_weighted \
  --epochs 50
```

**适用**: 快速验证流程

---

## 📊 **预期效果**

| 方法 | Interictal F1 | Preictal F1 | Ictal F1 | Macro F1 |
|------|--------------|-------------|----------|----------|
| 直接训练（无优化） | 0.99 | 0.00 | 0.00 | 0.33 ❌ |
| 加权损失 | 0.95 | 0.45 | 0.30 | 0.57 ✅ |
| 焦点损失 | 0.94 | 0.58 | 0.42 | 0.65 ✅✅ |
| 焦点损失 + SMOTE | 0.92 | 0.62 | 0.48 | 0.67 ✅✅✅ |

---

## 🔍 **调参建议**

### 焦点损失 Gamma 参数
```python
# 不平衡程度 10:1  → gamma=1.0
# 不平衡程度 100:1 → gamma=2.0  ⭐ 推荐
# 不平衡程度 1000:1 → gamma=3.0
```

### 类别权重调整
```python
# 自动计算（推荐）
class_weights = compute_class_weight('balanced', ...)

# 手动调整（微调）
class_weights = [0.5, 100, 1000]  # 给发作期更高权重
```

---

## 📚 **参考资料**

1. **Focal Loss 论文**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
2. **SMOTE 论文**: [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
3. **癫痫预测综述**: [Deep Learning for Seizure Prediction](https://ieeexplore.ieee.org/document/9345488)

---

## 💡 **常见问题**

### Q: 为什么 Accuracy 高但 F1 很低？
**A**: 模型只学会了预测多数类。用 F1、AUPRC 代替 Accuracy。

### Q: 过采样后训练时间变长？
**A**: 正常现象。可以先用少量数据测试，确认有效后再全量训练。

### Q: 焦点损失和加权损失能同时用吗？
**A**: 可以！焦点损失内部已经支持 alpha（类别权重）参数。

### Q: 少数类样本太少（<10个）怎么办？
**A**: 
1. 收集更多数据（最优）
2. 改为异常检测问题（One-Class SVM）
3. 使用迁移学习（预训练模型）
