# 数据处理
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

# 杂项
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
sns.set(palette='muted', style='whitegrid')
np.random.seed(13154)

train = pd.read_csv('C:/Users/lmx/Desktop/满意度/train.csv')
test = pd.read_csv('C:/Users/lmx/Desktop/满意度/test.csv')
print('训练集样本数为 %i, 变量数为 %i' % (train.shape[0],train.shape[1]))
print('测试集样本数为 %i, 变量数为 %i' % (test.shape[0],test.shape[1]))


i = 0
for col in train.columns:
    if train[col].var() == 0:
        i += 1
        del train[col]
        del test[col]
print('%i 个特征具有零方差并且已被删除' % i)


#过滤稀疏特征
#函数numpy.percentile():百分位数是统计中使用的度量，表示小于这个值的观察值的百分比。
i=0
for col in train.columns:
    if np.percentile(train[col],99) == 0:
        i += 1
        del train[col]
        del test[col]
print('%i 个特征是稀疏的并且已被删除' % (i))


#获取所有列的两两组合
#来自 itertools 模块的函数 combinations(list_name, x) 将一个列表和数字 ‘x’ 作为参数，并返回一个元组列表，每个元组的长度为 ‘x’，其中包含x个元素的所有可能组合。
# 列表中元素不能与自己结合，不包含列表中重复元素
combinations = list(itertools.combinations(train.columns,2))
print(combinations[:20])

len(combinations)#11026
#删除重复特征，保留其一
remove = []
keep = []
for f1,f2 in combinations:
    if (f1 not in remove) & (f2 not in remove):
        if train[f1].equals(train[f2]):
            remove.append(f1)
            keep.append(f2)
train.drop(remove,axis=1,inplace=True)
test.drop(remove,axis=1,inplace=True)
print('%i 个特征是重复的,并且 %i个特征已被删除' % (len(remove)*2,len(remove)))
print('其中特征 %s被删除\n特征 %s 被保留下来' % (remove,keep))
del remove
del keep
del combinations

print('训练集缺失值数量和: %i' % (train.isnull().sum().sum()))
print('测试集缺失值数量和: %i' % (test.isnull().sum().sum()))

