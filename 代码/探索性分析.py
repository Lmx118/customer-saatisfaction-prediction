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



def countplot_target(df,h=500):
    '''
        :desc  绘制目标变量的频率分布，并输出满意客户和不满意客户的数量
        :param h:数据标签的附加高度
    '''
    plt.figure(figsize=(5,5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False#运行配置参数总的轴（axes）正常显示正负号（minus）
    ax = sns.countplot(x='TARGET',data=df)
     # ax.patches 表示条形图中的每一个矩形
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2,height + h,'{:1.2f}%'.format(height*100/df.shape[0]),ha='center')#指定文字显示的位置
    plt.title('TARGET变量的频率分布图')
    print('满意客户的数量为%i,不满意客户的数量为 %i' % (
        df[df['TARGET']==0].shape[0],
        df[df['TARGET']==1].shape[0]
    ))
    plt.show()

#定义绘制函数hisplot_comb
def hisplot_comb(col,train=train,test=test,size=(20,5),bins=20):
    '''
        绘制训练集和测试集某一特征的直方图
    '''
    plt.subplots(1,2,figsize=size)#分割界面为1行2列
    plt.subplot(121)
    plt.title('训练集特征{}的分布'.format(col))
    plt.ylabel('频数')
    plt.xlabel(col)
    plt.hist(train[col],bins=bins)

    plt.subplot(122)
    plt.title('测试集特征{}的分布'.format(col))
    plt.ylabel('频数')
    plt.xlabel(col)
    plt.hist(test[col],bins=bins)#bins：直方图的柱数，即要分的组数
    plt.show()

def valuecounts_plot(col,train=train,test=test):
    '''
        绘制训练集和测试集特定列的频数分布折线图，并输出出现百分比最高的前5个值和最低的前5个值
    '''
    plt.subplots(1,2,figsize=(15,6))
    plt.subplot(121)
    df = train[col].value_counts().sort_index()
    sns.lineplot(x=df.index,y=df.values)
    plt.title("%s的频数分布折线图" % (col))
    plt.ylabel('频数')

    plt.subplot(122)
    df = test[col].value_counts().sort_index()
    sns.lineplot(x=df.index,y = df.values)
    plt.title("%s的频数分布折线图" % (col))
    plt.ylabel('频数')

    plt.tight_layout()
    #tight_layout会自动调整子图参数，使之填充整个图像区域。
    # 这是个实验特性，可能在一些情况下不工作。它仅仅检查坐标轴标签、刻度标签以及标题的部分。
    plt.show()

    print("*"*100)
    print("训练集特征'%s'其值占比(top 5): " % (col))
    print("值\t 占比%")
    print((train[col].value_counts()*100/train.shape[0]).iloc[:5])
    print("*"*100)
    print("训练集特征'%s'其值占比(bottom 5): " % (col))
    print("值\t 占比%")
    print((train[col].value_counts()*100/train.shape[0]).iloc[-5:])

    print("测试集特征'%s'其值占比(top 5): " % (col))
    print("值\t 占比%")
    print((test[col].value_counts()*100/test.shape[0]).iloc[:5])
    print("*"*100)
    print("测试集特征'%s'其值占比(bottom 5): " % (col))
    print("值\t 占比%")
    print((test[col].value_counts()*100/test.shape[0]).iloc[-5:])

#定义绘图函数hisplot_target
def histplot_target(col,df=train,height=6,bins=20):
    '''
    :param col: 特征
    :param df: 数据集
    :param height: 附加高度
    :param bins: 柱子数量
    :return:
    '''
    sns.FacetGrid(data=df,hue='TARGET',height=height).map(plt.hist,col,bins=bins).add_legend()
    plt.title('特征%s在不同目标变量下的频数分布' % (col))
    plt.ylabel('频数')
    plt.show()

countplot_target(train,h=500)


# var3(Region)
np.array(sorted(train.var3.unique()))
print('共有%i个唯一值' % (len(np.array(sorted(train.var3.unique())))))

print("值\t  计数")
print((train['var3'].value_counts()[:5]))
print("值\t  占比%")
print(train['var3'].value_counts()[:5]/train.shape[0]*100)

train['var3'].replace(-999999,2,inplace=True)
test['var3'].replace(-999999,2,inplace=True)
countplot_target(train[train['var3'] == 2],h=20)
countplot_target(train[train['var3'] != 2],h=10)


# var15（Age）
print('var15 最小值为: %i,最大值为: %i' % (train['var15'].min(),train['var15'].max()))
#var15 最小值为: 5,最大值为: 105

hisplot_comb('var15')
#stats.percentileofscore  计算分数相对于分数列表的一个排名情况 第一个参数是分数列表第二个是分数
print("训练集中年龄在30岁以下的客户约占所有数据的 %.2f%%" % (stats.percentileofscore(train['var15'].values,30)))
print("测试集中年龄在30岁以下的客户约占所有数据的 %.2f%%" % (stats.percentileofscore(test['var15'].values,30)))
#训练集中年龄在30岁以下的客户约占所有数据的 56.15%
#测试集中年龄在30岁以下的客户约占所有数据的 56.58%

ax = histplot_target('var15',bins=10)
plt.figure(figsize=(6,6))
mask = train[train['TARGET']==1]
plt.hist(mask['var15'],color='orange')
plt.title('特征var15在target=1下的频数分布')
plt.xlabel('var15')
plt.show()

# 创建新特征用来判断客户是否小于23岁
for df in [train,test]:
    df['var15_below_23'] = np.zeros(df.shape[0],dtype=int)
    df.loc[df['var15'] < 23,'var15_below_23'] = 1#把var15列小于23的行记录中的var15_below_23的部分赋值为1

_,bins = pd.cut(train['var15'].values , 5,retbins=True)#retbins： 是否显示分箱的分界值。默认为False，当bins取整数时可以设置retbins=True以显示分界值，得到划分后的区间
print(_)

train['var15'] = pd.cut(train['var15'].values,bins,labels=False)
test['var15'] = pd.cut(test['var15'].values,bins,labels=False)
histplot_target('var15')

# var38(Mortgage values)
print('最小值是 %i,最大值为 %i' % (train['var38'].min(),train['var38'].max()))
sorted(train['var38'].unique())
train.var38.value_counts()

for i in np.arange(0,1.1,0.1):
    print('%i percentile : %i' % (i*100,np.quantile(train.var38.values,i)))

mask = train[train['var38'] <= np.quantile(train.var38.values,0.975)]
histplot_target('var38',df=mask,bins=20)

# 对数变换
mask['var38'] = np.log(mask.var38).values
histplot_target('var38',df=mask,bins=20)

for df in [train,test]:
    df['var38'] = np.log(df['var38']).values
histplot_target('var38',bins=20)

import re
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

k = pd.Series(f_keywords)
ax = sns.barplot(x=k.index,y=k.values)
plt.title('特征关键词前缀的频数分布')
plt.ylabel('频数')
plt.xlabel('关键词前缀')
plt.show()

col = 'imp_trans_var37_ult1'
print("训练集中 saldo_medio_var13_corto_ult3 最小值为：%i，最大值为：%i" % (train[col].min(), train[col].max()))
print("测试集中 saldo_medio_var13_corto_ult3 最小值为：%i，最大值为：%i" % (test[col].min(), test[col].max()))
# 绘制特定列的频数分布折线图
valuecounts_plot(train=train, test=test, col=col)

df = train[train[col] != 0]
df1 = test[test[col] != 0]
for data in [df, df1]:
    data.loc[data[col] != 0, col] = np.log(data.loc[data[col] != 0, col])
hisplot_comb(col, train=df, test=df1)

train.to_pickle('train.pkl')
test.to_pickle('test.pkl')
