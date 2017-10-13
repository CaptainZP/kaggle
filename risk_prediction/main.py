import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from divide_dataset import divide_dataset
from sklearn.ensemble import ExtraTreesClassifier
from preprocessing.missing_data_address import missing_data_statistics, delete_nacol, delete_nasample, missing_data_complement


features = pd.read_csv("data\\1entbase.csv")
na_info = missing_data_statistics(features)
print('缺失值信息:\n', na_info)
print(features.shape)
features = delete_nacol(features)
new_na_info = missing_data_statistics(features)
print('\n去高缺失行后信息:\n', new_na_info)
print(features.shape)
features = missing_data_complement(features)
print('\n补全缺失值后数据样例:\n', features.head())
print(features.shape)

print('\n-----------求train文件特征与标签---------------')
train_eid = pd.read_csv("data\\train.csv")['EID']
train_features = features[features['EID'].isin(train_eid)].drop(['EID', 'RGYEAR'], axis=1)  # EID对应正确
print('train_features number:{}'.format(len(train_features)))
print('train_features:\n', train_features.head(), '\n', train_features.tail())
train_targets = pd.read_csv("data\\train.csv").drop('EID', axis=1)

print('\n-----------划分测试集训练集---------------')
x_train, x_test, y_train, y_test = divide_dataset(train_features.values, train_targets.values)   # 划分数据集values转ndarray
print('划分后数据形状:', x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print('train number:{}, test number:{}'.format(len(y_train), len(y_test)))
# y_train = y_train.ravel()
# y_test = y_test.ravel()
# print('处理后数据形状:', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print('\n-----------特征重要性---------------')
forest = ExtraTreesClassifier(n_estimators=20, random_state=0)    # 组合决策树对特征重要性评级
train_features = train_features.values
train_targets = train_targets.values.ravel()
forest.fit(train_features, train_targets)
importances = forest.feature_importances_
print('future importance:', importances)

print('\n-----------测试集测试结果---------------')
# clf = SVC(probability=True)
# clf = SGDClassifier(loss='log')
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)   # 测试集预测正负结果数组
y_scores = clf.predict_proba(x_test)[:, 1]  # 估计为正例的概率
print('预测值:\n', y_pred[0:10])
print('真实值:\n', y_test[0:10].T)
print('预测为正的概率:\n', y_scores[0:10])
auc = roc_auc_score(y_test, y_scores)
print('auc = {}'.format(auc))

print('\n-----------求evaluation_public文件特征---------------')
test_eid = pd.read_csv("data\\evaluation_public.csv")['EID']
test_features = features[features['EID'].isin(test_eid)].drop(['EID', 'RGYEAR'], axis=1)
print('test_features number:{}'.format(len(test_features)))
print('test_features:\n', test_features.head(), '\n', test_features.tail())

print('\n-----------预测evaluation_public标签---------------')
y_pred2 = clf.predict(test_features)
y_scores2 = clf.predict_proba(test_features)[:, 1]  # 估计为正例的概率

eid = pd.read_csv("data\\evaluation_public.csv").values
print(type(eid), type(y_pred2), type(y_scores2))
print(eid.shape, y_pred2.shape, y_scores2.shape)
y_pred2 = y_pred2.reshape((-1, 1))
y_scores2 = y_scores2.reshape((-1, 1))
print(eid.shape, y_pred2.shape, y_scores2.shape)
result_ndarray = np.concatenate((eid, y_pred2, y_scores2), axis=1)
time = time.time()
print(time)
np.savetxt('C:\\Users\\captainzp\\Desktop\\result.csv', result_ndarray, fmt=('%d', '%d', '%.4f'), delimiter=',')
# result_dataframe = pd.DataFrame(result_ndarray, columns=['EID', 'FORTARGET', 'PROB'])
# result_dataframe.to_csv('result.csv', index=False, float_format='%.4f')

