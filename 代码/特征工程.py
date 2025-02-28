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


train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')
X_train = train.copy()
X_test = test.copy()
print(X_train.shape,X_test.shape)

def add_feature_no_zeros(train=X_train,test = X_test):
    #构造新特征，表示每行样本中143个特征取值为0和非零的出现次数
    col = [k for k in train.columns if k != 'TARGET']
    for df in [train,test]:
        df['no_zeros'] = (df.loc[:,col] != 0).sum(axis=1).values
        df['no_nonzeros'] = (df.loc[:,col] == 0).sum(axis=1).values


def add_feature_no_zeros_keyword(keyword,train=X_train,test=X_test):
    col = [k for k in train.columns if keyword in k]
    # for k in col:
    for df in [train,test]:
        df['no_zeros_'+keyword] = (df.loc[:,col] != 0).sum(axis=1).values
        df['no_nonzeros_'+keyword] = (df.loc[:,col] == 0).sum(axis=1).values


print([col for col in train.columns if col[:3] == 'var'])
f_keywords = {col.split('_')[0] for col in train.columns if (len(col.split('_')) > 1) &~ ('var15' in col)}#&~:满足前面但不满足后面
print(f_keywords)

# 计算每种关键词前缀特征的计数
f_keywords = zip(f_keywords, np.zeros(len(f_keywords),dtype=int))
f_keywords = dict(f_keywords)
for key in f_keywords.keys():
    for col in train.columns:
        if key in col:
            f_keywords[key] += 1
print(f_keywords)


add_feature_no_zeros()
keywords = list(f_keywords.keys())
for k in keywords:
    add_feature_no_zeros_keyword(k)

def average_col(col,features,train=X_train,test=X_test):
    '''
    获取'col'特征中每一种唯一值的情况下feature特征的均值，并令其为新特征
    '''
    for df in [train,test]:
        unique_values = df[col].unique()

        for feature in features:
            #对每一个特征求他在指定特征col的每一个唯一值下的均值
            avg_value = []
            for value in unique_values:
                #对于每一个特征列col，求其每一种唯一值的情况下feature特征的均值
                avg = df.loc[df[col] == value,feature].mean()
                avg_value.append(avg)
            avg_dict = dict(zip(unique_values,avg_value))
            new_col = 'avg_'+ col + '_' + feature

            df[new_col] = np.zeros(df.shape[0])#新建新特征
            for value in unique_values:
                df.loc[df[col]==value,new_col] = avg_dict[value]
#含imp和saldo前缀的所有特征，不包括no_zeros_imp和no_zeros_saldo
features = [i for i in X_train.columns if (('imp' in i) | ('saldo' in i)) & ('no_zeros' not in i)]

#唯一值个数处于(50,210]之间的特征列
columns = [i for i in X_train.columns if (X_train[i].nunique() <= 210) & (X_train[i].nunique() > 50)]
len(features),len(columns)
for col in tqdm(columns):
    average_col(col,features)

print(X_train.shape,X_test.shape)



# 过滤冗余特征
def remove_corr_var(train=X_train,test=X_test,
                    target_threshold=10**-3,within_threshold=0.95):
    #删除与目标变量相关性低的特征，删除彼此之间相关性高的特征（保留一个）
    initial_feature = train.shape[1]
    corr = train.drop('ID',axis=1).corr().abs()
    corr_target = pd.DataFrame(corr['TARGET']).sort_values(by='TARGET')
    print('corr_target')
    print(corr_target)
    feat_df = corr_target[corr_target['TARGET']<=target_threshold]
    print('有 %i 个特征因为与目标变量TARGET的相关系数绝对值小于 %.3f而被删除' % (feat_df.shape[0],target_threshold))
    print('deleting...')
    for df in [train,test]:
        df.drop(feat_df.index,axis=1,inplace=True)
    print('已删除！')

    #删除彼此之间相关性高的特征(保留一个与TARGET相关性最高的特征)
    corr.sort_values(by='TARGET',ascending=False,inplace=True)#将相关矩阵每一行先按TARGET列降序排列
    corr = corr.reindex(columns=corr.index)
    corr.drop('TARGET',axis=1,inplace=True)#删除target列
    corr.drop('TARGET',axis=0,inplace=True)
    corr.drop(feat_df.index,axis=1,inplace=True)#删除feat_df中特征在corr表corr表里的列
    corr.drop(feat_df.index,inplace=True)
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool_))  # 获取相关矩阵的上三角
    column = [col for col in upper.columns if any(upper[col] > within_threshold)]  # 获取与特征之一高度相关的所有列
    print("有 %i 个特征与另一个特征高度相关且相关系数为 %.3f 及以上而被删除" % (len(column), within_threshold))
    print("删除中.........")
    for df in [train, test]:
        df.drop(column, axis=1, inplace=True)
    print("已删除！")

    print("特征数从 %i 个变成 %i 个，其中 %i 个特征已被删除" %
          (initial_feature, test.shape[1], initial_feature - test.shape[1]))

remove_corr_var(train=X_train,test=X_test,target_threshold=10**-3,within_threshold=0.95)


# 保存为P文件，方便后续调用
X_train.to_pickle('X_train.pkl')
X_test.to_pickle('X_test.pkl')

# 读取上述P文件
X_train = pd.read_pickle('X_train.pkl')
X_test = pd.read_pickle('X_test.pkl')

print(X_train.shape,X_test.shape)

