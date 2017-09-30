from houseprice_predict.get_dataframe import get_dataframe
from houseprice_predict.data_visualizing.var_dtype_count import var_dtype_count
from houseprice_predict.data_visualizing.plot_figure import plot_scatter
from houseprice_predict.data_visualizing.plot_figure import plot_box
from houseprice_predict.data_visualizing.correlation_analysis import allvars_correlation
from houseprice_predict.data_visualizing.correlation_analysis import continuous_dependent_correlation
from houseprice_predict.data_visualizing.correlation_analysis import separated_dependent_correlation
from houseprice_predict.data_visualizing.collinearity_analysis import allvars_collinearity
from houseprice_predict.data_visualizing.collinearity_analysis import continuous_vars_collinearity
from houseprice_predict.data_visualizing.collinearity_analysis import separated_vars_collinearity


file = "C:\\Users\\captainzp\\Desktop\\houseprice_train.csv"
columns = ['销售日期', '销售价格', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数',
           '房屋评分', '建筑面积', '地下室面积', '建筑年份', '修复年份', '纬度', '经度']
df = get_dataframe(file, columns)

# ##############变量类型及各类型#############################
quantitative_len, qualitative_len = var_dtype_count(df)
print(df.get_dtype_counts())
print("数值变量个数: {}, 类型变量个数: {}".format(quantitative_len, qualitative_len))
print('------------------------------------')

# #############连续型数值变量与价格散点图####################
continuous_vars = ['销售日期', '房屋面积', '停车面积', '建筑面积', '地下室面积', '建筑年份', '纬度', '经度']
price = '销售价格'
for var in continuous_vars:
    plot_scatter(df, var, price)

# #############离散型数值变量与价格箱型图####################
separated_vars = ['销售日期', '卧室数', '浴室数', '楼层数', '房屋评分']
price = '销售价格'
for var in separated_vars:
    plot_box(df, var, price)

# #############伪连续伪离散变量处理##########################
# df['销售日期'] = df[df['销售日期'] > 20150101]
# plot_box(df, '销售日期', '销售价格')
# df['修复年份'] = df[df['修复年份'] > 0]

# ############变量与价格的相关性分析###################
continuous_columns = ['销售日期', '房屋面积', '停车面积', '建筑面积', '地下室面积', '建筑年份', '纬度', '经度']
separated_columns = ['卧室数', '浴室数', '楼层数', '房屋评分', '修复年份']
target_column = '销售价格'
df = get_dataframe(file, cols=columns)
# 分析所有变量
allvars_correlation(df)
# plot_scatter(df, '修复年份', '销售价格')
# df['房屋年龄'] = df['销售日期'] - df['修复年份']*10000
# index = df[df['修复年份'] == 0].index
# df['房屋年龄'][index] = df['销售日期'][index] - df['建筑年份'][index]*10000
# plot_scatter(df, '房屋年龄', '销售价格')
# 连续型变量
continuous_dependent_correlation(df, continuous_columns, target_column)
# 离散型变量
separated_dependent_correlation(df, separated_columns, target_column)

# ############变量的共线性分析###################
continuous_columns = ['销售日期', '房屋面积', '停车面积', '建筑面积', '地下室面积', '建筑年份', '纬度', '经度']
separated_columns = ['卧室数', '浴室数', '楼层数', '房屋评分', '修复年份']
df = get_dataframe(file, cols=columns).drop('销售价格', axis=1)
allvars_collinearity(df)
continuous_vars_collinearity(df, continuous_columns)
separated_vars_collinearity(df, separated_columns)
