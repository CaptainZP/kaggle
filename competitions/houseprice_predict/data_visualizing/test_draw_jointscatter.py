from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from houseprice_predict.get_dataframe import get_features
from houseprice_predict.get_target import get_target

a = [1, 2, 3, 4]
b = [1, 2, 3, 4]
c = [1, 10, 30, 50]
file = "C:\\Users\\captainzp\\Desktop\\houseprice_train.csv"
feature_columns = ['纬度', '经度']
target_column = ['销售价格']
features = get_features(file, cols=feature_columns, date_col=False)
targets = get_target(file, target_column)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(features['纬度'], features['经度'], targets, cmap='Greens')
plt.show()



