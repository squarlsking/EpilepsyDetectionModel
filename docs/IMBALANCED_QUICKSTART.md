# 🎯 类别不平衡快速指南

## 📊 **你的数据情况**

```
间歇期（label=0）: 14,777 样本 (99.13%)
预发作期（label=1）: 120 样本 (0.80%)
发作期（label=2）: 10 样本 (0.07%)
```

**不平衡比例**: 约 **1477:12:1** （极度不平衡！）

---

## 🚀 **推荐方案（从易到难）**

### ✅ **方案 1: 焦点损失（最推荐）**

```bash
# 一键训练
python scripts/train_xcm_imbalanced.py \
  --csv data/processed/test_with_seizures.csv \
  --output models/xcm_focal \
  --use-focal-loss \
  --epochs 50
```

**特点**:
- ✅ 自动关注困难样本
- ✅ 适合极度不平衡（99:1）
- ✅ 经典算法（YOLO、RetinaNet 都用）
- ⏱️ 训练时间: 正常

**预期效果**: Macro F1 ~ 0.60-0.70

---

### ⚡ **方案 2: 加权损失（快速测试）**

```bash
python scripts/train_xcm_imbalanced.py \
  --csv data/processed/test_with_seizures.csv \
  --output models/xcm_weighted \
  --epochs 30
```

**特点**:
- ✅ 实现简单
- ✅ 训练速度最快
- ⚠️ 效果稍弱于焦点损失

**预期效果**: Macro F1 ~ 0.50-0.60

---

### 🔥 **方案 3: 焦点损失 + SMOTE（数据少时）**

```bash
python scripts/train_xcm_imbalanced.py \
  --csv data/processed/test_with_seizures.csv \
  --output models/xcm_focal_smote \
  --use-focal-loss \
  --use-smote \
  --epochs 50
```

**特点**:
- ✅ 增加少数类样本
- ✅ 适合样本量 < 5,000
- ⚠️ 可能过拟合
- ⏱️ 训练时间: 较长

**预期效果**: Macro F1 ~ 0.65-0.75

---

## 📈 **效果对比（预测）**

| 方法 | Interictal | Preictal | Ictal | Macro F1 |
|------|-----------|----------|-------|----------|
| 无优化 | 0.99 | 0.00 | 0.00 | 0.33 ❌ |
| 加权损失 | 0.95 | 0.45 | 0.30 | 0.57 ✅ |
| **焦点损失** | 0.94 | 0.58 | 0.42 | **0.65** ⭐ |
| 焦点+SMOTE | 0.92 | 0.62 | 0.48 | 0.67 ✅✅ |

---

## 🎓 **评估指标（重要！）**

### ❌ **不要只看 Accuracy**
```
模型全预测为 0 → Accuracy = 99%  # 看起来很高，实际啥也没学
```

### ✅ **应该看的指标**

1. **F1 Score（每个类别）**
   ```python
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ```

2. **Macro F1（整体效果）**
   ```python
   Macro F1 = (F1_class0 + F1_class1 + F1_class2) / 3
   ```

3. **混淆矩阵**
   ```
                Pred 0  Pred 1  Pred 2
   True 0 (间歇)   4500     50       0
   True 1 (预发)     30     60       10
   True 2 (发作)      2      3        5
   ```

4. **AUPRC（最重要！）**
   - PR 曲线下面积
   - 适合极度不平衡

---

## 🔧 **调参指南**

### Focal Loss Gamma
```python
# 根据不平衡程度选择
不平衡 10:1   → gamma=1.0
不平衡 100:1  → gamma=2.0  ⭐ 推荐
不平衡 1000:1 → gamma=3.0
```

### 训练轮数
```bash
# 快速测试
--epochs 10

# 正常训练
--epochs 50  ⭐ 推荐

# 充分训练
--epochs 100
```

### 批大小
```bash
# GPU 显存小
--batch-size 16

# 正常
--batch-size 64  ⭐ 推荐

# GPU 显存大
--batch-size 128
```

---

## 💡 **常见问题**

### Q: 为什么 Macro F1 只有 0.6？
**A**: 
- 少数类样本太少（label=2 只有 10 个）
- 这已经是不错的结果！
- 可尝试：
  1. 收集更多发作期数据
  2. 使用 SMOTE 过采样
  3. 改为二分类（合并 label=1 和 label=2）

---

### Q: 训练时 loss 不下降？
**A**:
1. 检查学习率（lr_find 自动寻找）
2. 增加训练轮数
3. 降低 focal_gamma（如 1.5）

---

### Q: 过采样后内存不足？
**A**:
1. 不使用 SMOTE（去掉 --use-smote）
2. 减小批大小（--batch-size 16）
3. 只对训练集过采样，不对验证集

---

## 🎯 **完整流程示例**

```bash
# 1. 处理数据（前 10 个受试者）
python scripts/prepare_dataset_xcm.py \
  --data-dir data/raw/ds005873-download \
  --output data/processed/seizure_10sub.csv \
  --max-subjects 10

# 2. 训练模型（焦点损失）
python scripts/train_xcm_imbalanced.py \
  --csv data/processed/seizure_10sub.csv \
  --output models/xcm_focal_10sub \
  --use-focal-loss \
  --epochs 50 \
  --batch-size 64

# 3. 查看结果
ls models/xcm_focal_10sub/
# model.pkl              - 训练好的模型
# confusion_matrix.png   - 混淆矩阵可视化
# results.json           - 详细评估指标
```

---

## 📚 **更多细节**

详细文档: [`docs/IMBALANCED_LEARNING.md`](./IMBALANCED_LEARNING.md)

- 焦点损失原理
- SMOTE 过采样机制
- 评估指标详解
- 论文参考

---

## ✨ **总结**

1. **首选方案**: 焦点损失（--use-focal-loss）
2. **评估指标**: F1、AUPRC（不看 Accuracy）
3. **数据不足**: 加上 --use-smote
4. **预期效果**: Macro F1 = 0.6-0.7（已经不错！）

🚀 **开始训练吧！**
