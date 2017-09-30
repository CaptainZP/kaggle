import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from houseprice_predict.get_target import get_target
from houseprice_predict.get_dataframe import get_dataframe

plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
file = "C:\\Users\\captainzp\\Desktop\\kc_train.csv"

feature_columns = ['销售价格', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数', '房屋评分', '建筑面积',
                   '地下室面积', '建筑年份', '修复年份', '纬度', '经度']
target_column = ['销售价格']
df = get_dataframe(file, feature_columns)
target = get_target(file, target_column)
# print(target.head(10))

print(df.head(3))
df['地下室面积'] = np.log1p(df['地下室面积'])
print(df.head(3))
# df['地下室面积'] = np.expm1(df['地下室面积'])
print(df.head(3))
res = stats.probplot(df['卧室数'], plot=plt)
# sns.distplot(df['地下室面积'], fit=stats.norm)   # 价格直方图,房价并不服从正态分布

# target['销售价格'] = np.log1p(target['销售价格'])   # 对数据取对数取消正偏性log/ log1p
# res = stats.probplot(target['销售价格'], plot=plt)   # 概率图可以发现，数据具有明显的正偏性，因此可采用对数来缓解这种趋势
# bb = target[target['地下室面积'] > 0]
# bb = np.log(bb)
# sns.distplot(bb, fit=stats.norm)
# # res = stats.probplot(bb['地下室面积'], plot=plt)
# plt.show()

# 查看其斜度skewness和峭度kurtosis，这是很重要的两个统计量
# print('skewness: {0}, kurtosis: {1}'.format(target['销售价格'].skew(), target['销售价格'].kurt()))


plt.show()
