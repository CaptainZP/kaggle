import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from houseprice_predict.get_dataframe import get_dataframe
from houseprice_predict.get_target import get_target

file = "C:\\Users\\captainzp\\Desktop\\houseprice_train.csv"
feature_columns = ['销售日期', '销售价格', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数', '房屋评分', '建筑面积',
                   '地下室面积', '建筑年份', '修复年份', '纬度', '经度']
target_column = ['销售价格']
features = get_dataframe(file, cols=feature_columns)
targets = get_target(file, target_column)

# 统计缺失值
missing = features.isnull().sum()
missing.sort_values(inplace=True, ascending=False)
# missing = missing[missing > 0]   # 确定有缺失值的列
types = features[missing.index].dtypes
percent = (features[missing.index].isnull().sum()/features[missing.index].isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percent, types], axis=1, keys=['missTotal', 'Percent', 'Types'])
missing_data.sort_values('missTotal', ascending=False, inplace=True)
print(missing_data)
missing.plot.bar()   # 打印缺失数量的柱状图
plt.show()


