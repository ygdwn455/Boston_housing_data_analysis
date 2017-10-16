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
boston_house = pd.read_csv('./data/housing.data', header=None, sep='\s+')
boston_house.columns = columns

print(boston_house.head(3))

"""
线性回归(Linear Regression)：
  假定特征与因变量是线性相关

岭回归(Ridge Regression)：
  在线性回归的损失函数上加上一个L2惩罚项，可以减小过拟合

Lasso回归：
  在线性回归的损失函数上加上一个L1惩罚项，可以使特征的权重降为0

ElasticNet回归：
  整合岭回归与Lasso回归

多项式回归(Polynomial Regression)：
  作用是可以拟合非线性数据

...
"""

# 2. 特征之间关系
import matplotlib.pyplot as plt
import seaborn
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'  # 字体


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
'''
  皮尔逊相关系数 r ：使用两特征的协方差除以其标准差的乘积
  如果 r=1, 表示两个特征完全正相关；如果 r=-1, 表示两个特征完全负相关。
  numpy.corrcoef : 计算相关系数
  seaborn.heatmap : 绘制相关系数对应的热度图
'''
import numpy as np
p_coef = np.corrcoef(boston_house.values.T)
print(p_coef)

#seaborn.heatmap(p_coef, annot=True, annot_kws={'size':5}, yticklabels=columns, xticklabels=columns)

#plt.savefig('特征间皮尔逊相关系数.png', dpi=300)
#plt.show()

X_rm = boston_house[['RM']].values
y_medv = boston_house['MEDV'].values

from sklearn.linear_model import LinearRegression
"""
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

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
'''
RANSAC是“RANdom SAmple Consensus（随机抽样一致）”的缩写。
它可以从一组包含“局外点”的观测数据集中，通过迭代方式估计数学模型的参数。
'''
from sklearn.linear_model import RANSACRegressor
lr = LinearRegression()
ransac = RANSACRegressor(base_estimator=lr, min_samples=50, residual_threshold=5, residual_metric=lambda x:np.sum(np.abs(x), axis=1), max_trials=500, random_state=0)
ransac.fit(X_rm, y_medv)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)  # 取反
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










