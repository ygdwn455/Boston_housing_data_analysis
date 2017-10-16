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

# 1. 加载数据集
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
houses = pd.read_table('./data/housing.data', header=None, sep='\s+')
houses.columns= columns

print(houses.head())

import matplotlib.pyplot as plt
import seaborn

"""
#seaborn.set(style='whitegrid')
seaborn.pairplot(houses[columns], size=2.5)
plt.savefig('特征相关性.png')
plt.show()
"""
import numpy as np

corr_coef = np.corrcoef(houses[columns].values.T)
seaborn.set(font_scale=1.0)
hm = seaborn.heatmap(corr_coef, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':5}, yticklabels=columns, xticklabels=columns)
plt.savefig('特征相关系数.png', dpi=300)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X_rm = houses[['RM']].values
y_medv = houses['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_rm_std = sc_x.fit_transform(X_rm)
y_medv_std = sc_y.fit_transform(y_medv)
lr = LinearRegression()
lr.fit(X_rm, y_medv)
print(lr.score(X_rm, y_medv))

plt.scatter(X_rm, y_medv, c='blue', edgecolor='black')
plt.plot(X_rm, lr.predict(X_rm), color='red')
plt.xlabel('每户房间数')
plt.ylabel('自住房屋拥有住房价格')
plt.show()

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, residual_metric=lambda x: np.sum(np.abs(x), axis=1), residual_threshold=5.0, random_state=0)
ransac.fit(X_rm, y_medv)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.scatter(X_rm[inlier_mask], y_medv[inlier_mask], c='blue', marker='o', label='内点')
plt.scatter(X_rm[outlier_mask], y_medv[outlier_mask], c='red', marker='s', label='异常点')
x_ = np.arange(3, 10, 1)
y_ = ransac.predict(x_[:, np.newaxis])
plt.plot(x_, y_, color='black')
plt.xlabel('每户房间数')
plt.ylabel('自住房屋拥有住房价格')
plt.legend(loc='upper left')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)






























































