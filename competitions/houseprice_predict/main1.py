import math
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from houseprice_predict.learning_curve import plot_learning_curve
from houseprice_predict.validation_curve import plot_validation_curve
from houseprice_predict.get_dataframe import get_dataframe
from houseprice_predict.get_target import get_target
from houseprice_predict.divide_dataset import divide_dataset
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# add this line to test git
# add again

# ##### 获取feature和target #####
file = "C:\\Users\\captainzp\\Desktop\\kc_train.csv"
feature_columns = ['销售日期', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数',
                   '房屋评分', '建筑面积', '建筑年份', '纬度', '经度']
target_column = ['销售价格']
features = get_dataframe(file, cols=feature_columns)
targets = get_target(file, col=target_column)


cols = ['房屋评分', '销售日期']
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(features[c].values))
    features[c] = lbl.transform(list(features[c].values))

features['房屋评分'] = features['房屋评分'].astype(str)
features['卧室数'] = features['卧室数'].astype(str)
features['浴室数'] = features['浴室数'].astype(str)
features['楼层数'] = features['楼层数'].astype(str)
features['销售日期'] = pd.Series.floordiv(features['销售日期'], 100)
features['销售日期'] = features['销售日期'].astype(str)

targets['销售价格'] = np.log1p(targets['销售价格'])
features['房屋面积'] = np.log1p(features['房屋面积'])
features['停车面积'] = np.log1p(features['停车面积'])
features['建筑面积'] = np.log1p(features['建筑面积'])
features = pd.get_dummies(features)
print('-------------')
print(features.head(2))



# shape
print('Shape all_data: {}'.format(features.shape))
print(features.head(5))

# numeric_feats = features.dtypes[features.dtypes != "object"].index
# print(numeric_feats)
# # Check the skew of all numerical features
# skewed_feats = features[numeric_feats].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew:', skewed_feats})
# skewness.head(5)


x_train, x_test, y_train, y_test = divide_dataset(features, targets)   # 划分数据集


def rmse_cv(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse = np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# lasso = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', linear_model.LinearRegression())])
# lasso = linear_model.Ridge(alpha=0.005)
lasso = make_pipeline(RobustScaler(), linear_model.Lasso(alpha=0.005, random_state=1))
score = rmse_cv(lasso)
print("\nLasso score: {} ({})\n".format(score.mean(), score.std()))


def rmse(y, y_pred):
    return (mean_squared_error(y, y_pred))/10000

lasso.fit(x_train.values, y_train)
stacked_train_pred = lasso.predict(x_train.values)
print(rmse(np.expm1(y_train), np.expm1(stacked_train_pred)))
stacked_pred = lasso.predict(x_test.values)
print(rmse(np.expm1(y_test), np.expm1(stacked_pred)))

file2 = "C:\\Users\\captainzp\\Desktop\\kc_test.csv"
features2 = get_dataframe(file2, cols=feature_columns)
features2['房屋面积'] = np.log1p(features2['房屋面积'])
features2['停车面积'] = np.log1p(features2['停车面积'])
features2['建筑面积'] = np.log1p(features2['建筑面积'])
features2['销售日期'] = pd.Series.floordiv(features2['销售日期'], 100)
test_pred = lasso.predict(features2.values)
np.savetxt("C:\\Users\\captainzp\\Desktop\\kc_result.csv", np.expm1(test_pred), format('%d'))


