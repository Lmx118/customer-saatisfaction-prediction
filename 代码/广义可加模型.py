from pygam import GAM
from pygam import s, te
from pygam import LinearGAM
from pygam import LogisticGAM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

#加载数据集
dataset = 'Normal'
train = pd.read_pickle('./datasets/train_normal.pkl')
test = pd.read_pickle('./datasets/test_normal.pkl')
X_train = train[['var3', 'var15', 'var38']]
y_train = train['TARGET'].values
X_test = test.drop('ID',axis=1)
test_id = test['ID']
del train,test

#划分数据集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.15)
print(X_train.shape, X_val.shape, X_test.shape)
# 定义广义可加模型
gam = LogisticGAM(s(0) + s(1) + s(2))

# 拟合模型
gam.fit(X_train, y_train)

# 预测
predictions = gam.predict(X_val)

# 查看模型摘要
print(gam.summary())

# 定义广义可加模型
gam = LinearGAM(s(0) + s(1) + s(2))

# 拟合模型
gam.fit(X_train, y_train)

# 预测
predictions = gam.predict(X_val)

# 查看模型摘要
print(gam.summary())