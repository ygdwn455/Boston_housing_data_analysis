# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:12:26 2017

@author: XiaoY

波士顿房屋数据分析

数据集
1. CRIM : 城镇人均犯罪率
2. ZN : 占地面积超过2.5万平方呎的住宅用地比例
3. INDUS : 每个城镇非零售商业用地的比例
4. CHAS : 查尔斯河虚拟变量(如果靠近河岸用1表示；否则用0表示)
5. NOX : 一氧化氮浓度
6. RM : 每户房间数
7. AGE : 在1940年之前建造的自用房屋的比例
8. DIS : 与五个波士顿劳动力聚集区的加权距离
9. RAD : 与辐射式公路接近指数
10. TAX : 每1万美元的全值财产税
11. PTRATIO : 学生/教师比例
12. B : 1000(Bk - 0.63)^2, 其中Bk是城镇的黑人比例
13. LSTAT : 低社会地位人口的比例
14. MEDV : 自住房屋拥有住房价格的中位数
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# 1. 加载并处理数据
# header=None 表示不使用表头，默认第一行为表头
# sep='\s+'，指定空格作为分隔符使用
boston_house = pd.read_csv('./data/housing.data', header=None, sep='\s+')
boston_house.columns = columns

print(boston_house.head(3))

"""
线性回归：
  假定特征与因变量是线性相关

岭回归：
  在线性回归的损失函数上加上一个L2惩罚项，可以减小过拟合

Lasso回归：
  在线性回归的损失函数上加上一个L1惩罚项，可以使特征的权重降为0

ElasticNet回归：
  整合岭回归与Lasso回归

多项式回归：
  作用是可以拟合非线性数据

...
"""

# 2. 特征之间关系
import matplotlib.pyplot as plt
import seaborn
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'


"""
seaborn.pairplot(boston_house, size=2.5)

plt.savefig('特征关系.png')
plt.show()
"""
# seaborn.boxplot(boston_house.MEDV)

"""
X, Y
x的方差 = sum((xi-x)*(xi-x))/n
x,y协方差 = sum((xi-x)*(yi-y))/n
p = 
"""
import numpy as np
p_coef = np.corrcoef(boston_house.values.T)
print(p_coef)

#seaborn.heatmap(p_coef, annot=True, annot_kws={'size':5}, yticklabels=columns, xticklabels=columns)

#plt.savefig('特征间皮尔逊相关系数.png', dpi=300)
#plt.show()

X_rm = boston_house[['RM']].values
y_medv = boston_house['MEDV'].values

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
"""
def plot_lr(X, y, model, c='red'):
    model.fit(X, y)
    plt.scatter(X, y, c='blue', edgecolor='black')
    plt.plot(X, model.predict(X), color=c)
    plt.xlabel('房间数')
    plt.ylabel('房价')
    plt.show


plot_lr(X_rm, y_medv, LinearRegression())
plot_lr(X_rm, y_medv, Ridge(), c='cyan')
plot_lr(X_rm, y_medv, Lasso(), c='black')
"""
"""
from sklearn.linear_model import RANSACRegressor
lr = LinearRegression()
ransac = RANSACRegressor(base_estimator=lr, min_samples=50, residual_threshold=5, residual_metric=lambda x:np.sum(np.abs(x), axis=1), max_trials=500, random_state=0)
ransac.fit(X_rm, y_medv)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
X_rm_in = X_rm[inlier_mask]
y_medv_in = y_medv[inlier_mask]
X_rm_out = X_rm[outlier_mask]
y_medv_out = y_medv[outlier_mask]
plt.scatter(X_rm_in, y_medv_in, c='blue', marker='o', label='内点')
plt.scatter(X_rm_out, y_medv_out, c='red', marker='s', label='异常点')
plt.plot(X_rm, ransac.estimator_.predict(X_rm), color='black', label='拟合线')
plt.xlabel('房间数')
plt.ylabel('房价')
plt.legend(loc='upper left')
plt.show()
"""

from sklearn.preprocessing import PolynomialFeatures

X_lstat = boston_house[['LSTAT']].values
pf = PolynomialFeatures(degree=3)

X_lstat_pf = pf.fit_transform(X_lstat)
lr = LinearRegression()
lr.fit(X_lstat_pf, y_medv)

plt.scatter(X_lstat, y_medv, c='blue')
plt.scatter(X_lstat, lr.predict(X_lstat_pf), c='red', marker='s')
plt.xlabel('低地位人口比例')
plt.ylabel('房价')
plt.show()

# 残差=预测结果-真实值
y_pred = lr.predict(X_lstat_pf)
plt.scatter(X_lstat, y_pred-y_medv, c='blue')
plt.plot(range(0,40), [0]*40, 'r--')
plt.xlabel('低地位人口比例')
plt.ylabel('房价')
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_medv, y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_medv, y_pred)

# 决策树
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=None)
tree.fit(X_lstat, y_medv)
plt.scatter(X_lstat, y_medv, c='blue')
plt.scatter(X_lstat, tree.predict(X_lstat), c='red')
plt.xlabel('低地位人口比例')
plt.ylabel('房价')
plt.show()
y_pred = tree.predict(X_lstat)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_medv, y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_medv, y_pred)


from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

X_lstat_train, X_lstat_test, y_medv_train, y_medv_test = train_test_split(X_lstat, y_medv)

forest = RandomForestRegressor()
forest.fit(X_lstat_train, y_medv_train)
y_pred = forest.predict(X_lstat_test)
print(mean_squared_error(y_medv_test, y_pred))
print(r2_score(y_medv_test, y_pred))

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(boston_house.loc[:, ['RM', 'LSTAT', 'PTRATIO']], boston_house.iloc[:, -1], test_size=0.2, random_state=0)

pipe = Pipeline([('mpl', MLPRegressor())])

param_grid = [{\
               'mpl__hidden_layer_sizes':[(100, ), (50, ), (20, 30)], \
               'mpl__activation':['identity', 'logistic', 'tanh', 'relu'], \
               'mpl__alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]\
               }]
gscv = GridSearchCV(pipe, param_grid=param_grid, cv=3)
gscv.fit(X_train, y_train)

print(gscv.best_estimator_.score(X_test, y_test))
