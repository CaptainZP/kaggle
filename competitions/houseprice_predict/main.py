import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from houseprice_predict.learning_curve import plot_learning_curve
from houseprice_predict.validation_curve import plot_validation_curve
from houseprice_predict.get_dataframe import get_dataframe
from houseprice_predict.get_target import get_target
from houseprice_predict.divide_dataset import divide_dataset
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree

logging.basicConfig(level=logging.INFO)
# ##### 获取feature和target #####
# data = pd.read_csv('C:\\Users\\captainzp\\Desktop\\houseprice_train.csv')
# prices = data['销售价格']
# features = data.drop('销售价格', axis = 1) #特征集是除了MEDV以外的其他数据的集合
file = "C:\\Users\\captainzp\\Desktop\\kc_train.csv"
# feature_columns = ['卧室数', '浴室数', '房屋面积', '停车面积', '楼层数', '房屋评分', '建筑面积',
#                    '地下室面积', '建筑年份', '修复年份', '纬度', '经度', '总房间数', '总面积']
feature_columns = ['卧室数', '房屋面积', '房屋评分', '纬度']
target_column = ['销售价格']   # date_column = ['销售日期']
features = get_dataframe(file, cols=feature_columns)
targets = get_target(file, col=target_column)
# features['房屋面积'] = np.log1p(features['房屋面积'])
targets['销售价格'] = np.log1p(targets['销售价格'])
# print('features type:', type(features))
# print('targets type:', type(targets))
# print('first 5 feature vectors:', features.head(1))   # 显示前几行
x_train, x_test, y_train, y_test = divide_dataset(features, targets)   # 划分数据集

# ##### SVR回归
# regr = SVR(kernel='rbf')
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### 最小二乘法
# regr = linear_model.LinearRegression()
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print('Coefficients:', regr.coef_)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### 岭回归
# regr = linear_model.RidgeCV(alphas=[0.1, 0.5, 1, 5, 10])   # 空参为自动寻找
# regr.fit(x_train, y_train)
# best_alpha = regr.alpha_
# print('best alpha:', best_alpha)
# y_pred = regr.predict(x_test)
#
# # regr = linear_model.Ridge(alpha=0.5)
# # regr.fit(x_train, y_train)
# # y_pred = regr.predict(x_test)
# print('Coefficients:', regr.coef_)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### 弹性网络
# regr = linear_model.ElasticNet()
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print('Coefficients:', regr.coef_)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### Lasso回归
# regr = linear_model.LassoCV(alphas=[0.1, 0.5, 1, 10])   # 空参为自动寻找
# regr.fit(x_train, y_train)
# best_alpha = regr.alpha_
# print('best alpha:', best_alpha)
# y_pred = regr.predict(x_test)
#
# # regr = linear_model.Lasso(alpha=0.5)
# # regr.fit(x_train, y_train)
# # y_pred = regr.predict(x_test)
# print('Coefficients:', regr.coef_)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### 最小角回归
# regr = linear_model.Lars()
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print('Coefficients:', regr.coef_)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### 贝叶斯岭回归
# regr = linear_model.BayesianRidge()
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print('Coefficients:', regr.coef_)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# # ##### 多项式回归
regr = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', linear_model.LinearRegression())])
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
# print('Coefficients:', regr.named_steps['linear'].coef_)
print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))
print('r2系数: %.2f' % r2_score(y_test, y_pred))   # 0至1，表示目标变量的预测值和实际值之间的相关程度平方的百分比,高好

# ##### SGD回归
# regr = linear_model.SGDRegressor(loss='squared_loss', penalty='l2')
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print('Coefficients:', regr.coef_)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### 决策树回归
# regr = tree.DecisionTreeRegressor(max_depth=8, random_state=0)
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print("特征重要性:", regr.feature_importances_)
# print("训练使用的特征数:", regr.n_features_)
# print("最大深度:", regr.max_depth)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### 混合模型
# regr = Pipeline([
#     ('ss', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=3, include_bias=True)),
#     ('linear', linear_model.ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
#                                          fit_intercept=False, max_iter=1e3, cv=3))
# ])
# regr.fit(x_train, y_train)
# y_pred = regr.predict(x_test)
# print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))

# ##### 画learning curve
# cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
# estimator = regr
# plot_learning_curve(estimator, 'Polynomial', features, targets, ylim=(0, 1), cv=cv)
# plt.show()


# ##### 画validation curve
# X, y = features.values, targets.values
# estimator = Pipeline([('poly', PolynomialFeatures()), ('linear', linear_model.LinearRegression())])
# # para_name = estimator.get_params().key()
# para_name = 'poly__degree'
# # print(type(para_name), para_name)
# para_range = [1, 2, 3]
# title = 'Ridge'
# xlabel_name = 'degree'
# plot_validation_curve(estimator, X, y, para_name, para_range, title, xlabel_name)

# print('预测值：', y_pred[0:10])
# print('真实值：', y_test.values[0:10])
file2 = "C:\\Users\\captainzp\\Desktop\\kc_test.csv"
features2 = get_dataframe(file2, cols=feature_columns)
features['房屋面积'] = np.log1p(features['房屋面积'])
test_pred = regr.predict(features2)
np.savetxt("C:\\Users\\captainzp\\Desktop\\kc_result.csv", test_pred, format('%d'))

# t = np.logspace(0, 50, len(y_pred))
# plt.figure(facecolor='w')
# plt.plot(t, y_test, 'r-', lw=2, label='真实值')
# plt.plot(t, y_pred, 'g-', lw=2, label='估计值')
# plt.legend(loc='best')
# plt.title('波士顿房价预测', fontsize=18)
# plt.xlabel('样本编号', fontsize=15)
# plt.ylabel('房屋价格', fontsize=15)
# plt.grid()
# plt.show()
