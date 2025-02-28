# 银行客户满意度预测系统 🏦

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)](https://xgboost.ai/)

基于Santander Bank客户数据，采用逻辑回归/随机森林/XGBoost构建二分类预测模型，实现客户满意度智能预警。

## 目录 📖
- [数据概况](#数据概况)
- [关键技术](#关键技术)
- [模型表现](#模型表现)
- [业务洞见](#业务洞见)
- [部署建议](#部署建议)

---

## 数据概况 📊

### 数据集特性
| 属性               | 描述                          |
|--------------------|-------------------------------|
| 来源               | Santander Bank (Kaggle竞赛数据) |
| 样本量             | 76,020条客户记录              |
| 特征维度           | 369个匿名特征                 |
| 类别分布           | 满意客户96.04% vs 不满意3.96% |

### 特征工程
```python
# 预处理流程
1. 删除零方差特征 → 消除无效维度
2. 过滤稀疏特征(零值占比>99%) → 去除低信息量维度
3. 去除重复特征 → 降低多重共线性
4. 异常值处理 → 修正var3的-9999999为缺失值
5. 特征变换 → 对imp/saldo类特征进行对数变换

## 关键技术 🧠

### 算法核心对比
| 模型        | 优势                          | 局限性                          | 关键参数配置                     |
|-------------|-------------------------------|---------------------------------|----------------------------------|
| **逻辑回归** | 可解释性强<br>训练速度快      | 线性假设限制<br>AUC最低(0.813)  | `penalty='l2'`<br>`C=0.001`      |
| **随机森林** | 抗过拟合<br>特征重要性分析     | 内存消耗大<br>训练耗时(254s)    | `n_estimators=1000`<br>`max_depth=1000` |
| **XGBoost**  | 最佳AUC(0.845)<br>并行加速     | 参数调优复杂                    | `learning_rate=0.01`<br>`max_depth=5` |

---

## 模型训练 ⚙️

### 通用训练流程
```python
# 数据预处理 -> 模型训练 -> 评估
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', CustomPreprocessor()),  # 含对数变换/缺失值处理
    ('classifier', XGBClassifier())          # 可替换为其他模型
])
pipeline.fit(X_train, y_train)

# XGBoost 最优参数配置（完整版）
xgb_params = {
    'objective': 'binary:logistic',    # 二分类任务
    'learning_rate': 0.01,             # 收缩步长防过拟合
    'max_depth': 5,                    # 树深限制
    'subsample': 0.9,                  # 行采样率
    'colsample_bytree': 0.5,           # 列采样率
    'scale_pos_weight': 24.3,          # 处理3.96%的类别不平衡
    'reg_alpha': 0.3,                  # L1正则化
    'gamma': 5                         # 分裂最小增益阈值
}


**设计说明**：
1. **关键技术**部分采用三维对比表，突出算法特性差异
2. **模型训练**包含可复用的代码模板与参数注释
3. **模型表现**通过多维指标+可视化+特征解释的三重验证
4. 使用`scale_pos_weight=24.3`显式处理类别不平衡问题（计算方式：负样本数/正样本数≈24.3）
5. 特征重要性表格直接关联业务决策，增强落地价值
