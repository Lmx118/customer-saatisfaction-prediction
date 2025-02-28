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

X_train = pd.read_pickle('X_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
print(X_train.shape, X_test.shape)


def apply_log1p(column, train=X_train, test=X_test):
    # 对数变换列特征
    tr = train.copy()
    te = test.copy()
    for df in [tr, te]:
        for col in column:
            df.loc[df[col] >= 0, col] = np.log1p(df.loc[df[col] >= 0, col].values)
    return tr, te


# 对所有最小值大于等于0的imp和saldo特征进行对数变换（var38在EDA中已经对数化，这里不再操作）
features = [i for i in X_train.columns if (('saldo' in i) | ('imp' in i)) & ((X_train[i].values >= 0).all())]
X_train_1, X_test_1 = apply_log1p(features)

X_train_1.to_pickle('X_train_1.pkl')
X_test_1.to_pickle('X_test_1.pkl')

# 选取唯一值的个数(2,10]的特征
cat_col = []
for col in X_train.columns:
    if (X_train[col].nunique() <= 10) & (col != 'TARGET') & (X_train[col].nunique() > 2):
        cat_col.append(col)
print("有 %i 个特征其唯一值数量(2,10]并使用它们创建独热编码和响应编码变量，同时删除原始特征" % (len(cat_col)))

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

print(cat_col)

def one_hot_encoding(col,train=X_train,test=X_test):
    #对训练集和测试集中的特征进行独热编码
    #一般的训练过程是enc.transform(df).toarray()，就是默认返回稀疏矩阵，然后便于查看所以转换为数组的形式，但是直接sparse=False就可以直接返回数组矩阵而不需要toarray()
    #默认情况下，handle_unknown = error，当遇到 transform时遇到fit中没有出现过的特征类别时，会直接报错
    #get_features:展示编码后的特征名，默认有个前缀是x0,x1...例如：array(['x0_PHD', 'x0_master', 'x1_A', 'x1_B', 'x1_C'], dtype=object)，但是这个x0也可以通过input_features参数指定
    #duplicated函数：找出df中的重复值
    ohe = OneHotEncoder( handle_unknown='ignore')
    ohe.fit(train[col])
    feature_names = list(ohe.get_feature_names_out(input_features=col))
    features = list(train.drop(col, axis=1).columns)
    features.extend(feature_names)

    # train
    df = train.copy()
    temp = ohe.transform(df[col])
    df.drop(col, axis=1, inplace=True)
    train = pd.DataFrame(hstack([df.values, temp]).toarray(), columns=features)
    train = train.loc[:, ~train.columns.duplicated(keep='first')]  # 删除重复行
    # test
    df = test.copy()
    temp = ohe.transform(df[col])
    df.drop(col, axis=1, inplace=True)
    features.remove('TARGET')
    test = pd.DataFrame(hstack([df.values, temp]).toarray(), columns=features)
    test = test.loc[:, ~test.columns.duplicated(keep='first')]

    return train, test


# 假设 cat_col 是您想要进行独热编码的特征列的名称
X_train_ohe, X_test_ohe = one_hot_encoding(cat_col)
X_train_1_ohe, X_test_1_ohe = one_hot_encoding(cat_col, X_train_1, X_test_1)
print(X_train_ohe.shape, X_test_ohe.shape, X_train_1_ohe.shape, X_test_1_ohe.shape)


def response_encoding_return(df, column, target, alpha=5000):
    """
    使用带有拉普拉斯平滑的响应编码到分类列column，并在训练、测试、验证数据集中转换相应的列。
    此函数用来训练出最优的参数alpha
    """
    unique_values = set(df[column].values)#求得所有唯一值
    dict_values = {}
    for value in unique_values:
        total = len(df[df[column] == value])
        sum_promoted = len(df[(df[column] == value) & df[target] == 1])
        dict_values[value] = np.round((sum_promoted + alpha) / (total + alpha * len(unique_values)), 2)
    return dict_values

# 寻找最好的alpha
def find_alpha(seed):
    random.seed(seed)
    ran_in = random.randint(0, 10)  # 随机生成0-9的整数
    col = [col for col in cat_col if X_train[col].nunique() > 3][ran_in]
    print('Feature: "%s"' % (col))
    for alpha in [100, 500, 1000, 2500, 5000, 10000]:
        print('for alpha %i:%s' % (alpha, response_encoding_return(X_train, col, "TARGET", alpha=alpha)))
find_alpha(seed=100)

def response_encoding(df,test_df,column,target='TARGET',alpha=5000):
    """
    在这里，我们使用带有拉普拉斯平滑的响应编码到分类列，并在训练、测试、验证数据集中转换相应的列。
    在这里，我们将重复每个类别的值 alpha 时间。
    """
    feature = column + '_1'
    feature_ = column + '_0'

    unique_values = set(df[column].values)
    dict_values = {} #存储target=1的响应编码值
    dict_values_ = {} #存储target=0的响应编码值
    for value in unique_values:
        total = len(df[df[column] == value])#此类别值在df中的个数
        #类别为某‘value’且目标变量为1时在df中的总个数
        sum_promoted = len(df[(df[column] == value) & (df[target] == 1)])
        sum_unpromoted = total - sum_promoted# 类别为某'vale'值且目标变量取0时在df中的总个数

        dict_values[value] = np.round((sum_promoted + alpha) / (total + alpha*len(unique_values)),2)
        dict_values[value] = np.round((sum_unpromoted + alpha) / (total + alpha*len(unique_values)),2)

    #假定了在某个字段中训练集出现的取值不完整，有些只在测试集中出现，这就是未知类别
    dict_values['unknown'] = 0.5#在训练集上观测不到的未知类别将被分配为0.5
    dict_values_['unknown'] = 0.5

    df[feature] = (df[column].map(dict_values)).values
    df[feature_] = (df[column].map(dict_values_)).values
    # print('dict_values: ')
    # print(dict_values)
    # print('dict_values_: ')
    # print(dict_values_)
    df.drop(column, axis=1, inplace=True)

    unique_values_test = set(test_df[column])
    #找出亮哥set中的不同元素并赋值为unknown
    test_df[column] = test_df[column].apply(lambda x: 'unknown' if x in (unique_values_test-unique_values) else x)
    test_df[feature] = (test_df[column].map(dict_values)).values
    test_df[feature_] = (test_df[column].map(dict_values_)).values
    test_df.drop(column, axis=1, inplace=True)

alpha = 100
X_train_re = X_train.copy()
X_test_re = X_test.copy()
X_train_1_re = X_train_1.copy()
X_test_1_re = X_test_1.copy()
for col in tqdm(cat_col):
    response_encoding(X_train_re, X_test_re, col, alpha=alpha)
    response_encoding(X_train_1_re, X_test_1_re, col, alpha=alpha)

print(X_train_re.shape, X_test_re.shape, X_train_1_re.shape, X_test_1_re.shape)

def stdzation(train,test):
    col = [i for i in train.columns if (i != 'TARGET') & (i != 'ID')]
    scaler = StandardScaler()
    train[col] = scaler.fit_transform(train[col])
    test[col] = scaler.transform(test[col])
datasets = [(X_train, X_test), (X_train_re, X_test_re), (X_train_ohe, X_test_ohe),
            (X_train_1, X_test_1), (X_train_1_re, X_test_1_re), (X_train_1_ohe, X_test_1_ohe)]
for train, test in datasets:
    stdzation(train, test)

datasets_labels = ["normal", 'normal_re',
                   "normal_ohe", "log", 'log_re', "log_ohe"]
print("不同数据集最终的特征数是：")
for i, (train, test) in enumerate(datasets):
    print("%s:\t%i" % (datasets_labels[i], test.shape[1]))

for i, (train, test) in enumerate(datasets):
    file=datasets_labels[i]+'.pkl'
    train.to_pickle('./datasets/train_'+file)
    train.to_pickle('./datasets/test_' + file)