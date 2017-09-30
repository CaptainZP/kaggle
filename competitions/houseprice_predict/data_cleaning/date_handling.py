from houseprice_predict.get_dataframe import get_features
from houseprice_predict.get_target import get_target
import pandas as pd
import numpy as np


file = "C:\\Users\\captainzp\\Desktop\\houseprice_train.csv"
feature_columns = ['销售日期']

feature = get_features(file, feature_columns)
print(type(feature.values))   # pandas的value是ndarray
feature1 = pd.to_datetime(feature['销售日期'], format='%Y%m%d')
print(feature1)
