import numpy as np
import pandas as pd
import random
import itertools
from scipy import stats
from scipy.sparse import hstack

# 数据可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 特征工程
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# 模型
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score, log_loss

import tqdm
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
sns.set(palette='muted', style='whitegrid')
np.random.seed(13154)


#加载数据集
dataset = 'Normal'
train = pd.read_pickle('./datasets/train_normal.pkl')
test = pd.read_pickle('./datasets/test_normal.pkl')
X_train = train.drop(['ID','TARGET'],axis=1)
y_train = train['TARGET'].values
X_test = test.drop('ID',axis=1)
test_id = test['ID']

del train,test

#划分数据集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.15)
print(X_train.shape, X_val.shape, X_test.shape)

# global i
i = 0

def plot_auc(y_true,y_pred,label,dataset = dataset):
    '''
    给出y_true和y_pred时绘制ROC曲线
    dataset:告诉我们使用了哪个数据集
    label:告诉我们使用了哪个模型，若label是一个列表，则绘制所有标签的所有ROC曲线
    '''
    if (type(label) != list) & (type(label) != np.array):
        print("\t\t %s on %s dataset \t\t \n" % (label, dataset))
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)  #-(ylog(p) + (1-y)log(1-p)))
        label_1 = label + ' AUC=%.3f' % (auc)

        # 绘制ROC曲线
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        sns.lineplot(x=fpr, y=tpr, label=label_1)
        x = np.arange(0, 1.1, 0.1)  # 绘制AUC=0.5的直线
        sns.lineplot(x=x, y=x, label="AUC=0.5")
        plt.title("ROC on %s dataset" % (dataset))
        plt.xlabel('False Positive Rate')
        plt.ylabel("True Positive Rate")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # 设置图例在图形外
        plt.show()
        print("在 %s 数据集上 %s 模型的 logloss = %.3f  AUC = %.3f" % (dataset, label, logloss, auc))

        # 创建结果数据框
        result_dict = {
            "Model": label,
            'Dataset': dataset,
            'log_loss': logloss,
            'AUC': auc
        }
        return pd.DataFrame(result_dict, index=[i])

    else:
        # 绘制ROC曲线

        for k, y in enumerate(y_pred):
            fpr, tpr, threshold = roc_curve(y_true, y)
            auc = roc_auc_score(y_true, y)
            label_ = label[k] + ' AUC=%.3f' % (auc)
            sns.lineplot(x=fpr, y=tpr, label=label_)

        x = np.arange(0, 1.1, 0.1)
        sns.lineplot(x=x, y=x, label="AUC=0.5")
        plt.title("Combined ROC")
        plt.xlabel('False Positive Rate')
        plt.ylabel("True Positive Rate")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2,borderaxespad=0.)
        plt.show()

def find_best_params(model, params, cv=10, n_jobs=-1, X_train=X_train):
    """
    使用随机搜索RandomizedSearchCV调参，返回最佳模型
    """
    random_cv = RandomizedSearchCV(model,
                                   param_distributions=params,
                                   scoring='roc_auc',
                                   n_jobs=n_jobs,
                                   cv=cv,
                                   verbose=2)
    random_cv.fit(X_train, y_train)
    print("最佳的AUC得分为：%.3f" % (random_cv.best_score_))
    print("最佳的参数为：%s" % (random_cv.best_params_))
    return random_cv.best_estimator_

# LogisticRegression初始化模型并设置参数
model_lr = LogisticRegression(class_weight='balanced',max_iter=1000)
params = {
    'penalty': ['l2', 'l1'],
    'C': [10.**i for i in np.arange(-3, 3, 1)],
    'fit_intercept': [True, False],
}
# 超参数寻优
find_best_params(model_lr, params)

# 拟合调参后的模型
model_lr = LogisticRegression(C=0.001, class_weight='balanced',max_iter=1000)
model_lr.fit(X_train, y_train)
#存储结果并绘制ROC曲线
labels = []
y_preds = []
y_pred = model_lr.predict_proba(X_val)[:,1]

label = "Logistic Regression"
labels.append(label)
y_preds.append(y_pred)
i = 0
result = plot_auc(y_val, y_pred, label, dataset)
result_df = result
del result

print(result_df)

# 随机森林
model_rf = RandomForestClassifier(class_weight='balanced')
params = {
    'n_estimators': [1000, 2000],
    'max_depth': [1000, 2000],
    'min_samples_split': [100, 500],
    'min_samples_leaf': [3, 5],
    'max_leaf_nodes': [100, 250]
}
find_best_params(model_rf, params, cv=3)

model_rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                                  criterion='gini', max_depth=2000, max_features='sqrt',
                                  max_leaf_nodes=250, max_samples=None,
                                  min_samples_leaf=3, min_samples_split=500,
                                  min_weight_fraction_leaf=0.0, n_estimators=2000,
                                  n_jobs=10, oob_score=False, random_state=42,
                                  verbose=0, warm_start=False)
model_rf.fit(X_train, y_train)
cc_model_rf = CalibratedClassifierCV(model_rf, cv='prefit')
cc_model_rf.fit(X_train, y_train)

y_pred = cc_model_rf.predict_proba(X_val)[:, 1]
label = "Random Forest Classifier"
labels.append(label)
y_preds.append(y_pred)

i += 1
result = plot_auc(y_val, y_pred, label)
result_df = result
del result
print(result_df)

# xgboost
model_xgb = xgb.XGBClassifier(n_jobs=-1,
                              nthread=-1,
                              scale_pos_weight=1.,
                              learning_rate=0.01,
                              colsample_bytree=0.5,
                              subsample=0.9,
                              objective='binary:logistic',
                              n_estimators=1000,
                              reg_alpha=0.3,
                              max_depth=5,
                              gamma=5,
                              random_state=42)

eval_metric = ['error', 'auc']
eval_set = [(X_train, y_train), (X_val, y_val)]
model_xgb.fit(X_train, y_train, eval_set=eval_set,
              eval_metric=eval_metric, early_stopping_rounds=50, verbose=20)

print(model_xgb.best_score, model_xgb.best_iteration)

# 基于迭代次数的调参曲线
results = model_xgb.evals_result_
auc_train = results['validation_0']['auc']
auc_val = results['validation_1']['auc']
fig, ax = plt.subplots(figsize=(10, 6))
epochs = len(auc_val)
ax.plot(range(0, epochs), auc_train, label='Train')
ax.plot(range(0, epochs), auc_val, label='Test')
ax.legend()
plt.title(model_xgb.__class__.__name__ + ' ' + 'AUC')
plt.ylabel('auc')
plt.show()
print("验证集上最大AUC：%.3f" % (max(auc_val)))
print("最优迭代次数epochs：%i" % (auc_val.index(max(auc_val))))


y_pred = model_xgb.predict_proba(X_val)[:, 1]
label = "XGBoost Classifier"
labels = []
y_preds = []
labels.append(label)
y_preds.append(y_pred)
result = plot_auc(y_val, y_pred, label, dataset)
i += 1
result = plot_auc(y_val, y_pred, label)
result_df = result
del result
print(result_df)