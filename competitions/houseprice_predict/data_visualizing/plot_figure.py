import matplotlib.pyplot as plt
import seaborn as sns
from houseprice_predict.get_dataframe import get_dataframe


def plot_scatter(df, xvar, yvar):
    '''
    画连续变量与因变量的散点图
    :param df: 数据的dataframe，dataframe
    :param xvar: 自变量，str
    :param yvar: 因变量，str
    :return:
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.subplots(1, 1, figsize=(15, 6))
    plt.scatter(df[xvar], df[yvar])
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.show()


def plot_box(df, xvar, yvar):
    '''
    画离散变量与因变量的箱型图
    :param df: 数据的datafrmae， dataframe
    :param xvar: 自变量，str
    :param yvar: 因变量，str
    :return:
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.subplots(1, 1, figsize=(15, 6))
    sns.boxplot(x=xvar, y=yvar, data=df)
    plt.show()

if __name__ == '__main__':
    file = "C:\\Users\\captainzp\\Desktop\\kc_train.csv"
    columns = ['销售日期', '销售价格', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数',
               '房屋评分', '建筑面积', '地下室面积', '建筑年份', '修复年份', '纬度', '经度']

    df = get_dataframe(file, columns)
    print(df.head(10))
    # df = df[df['修复年份'] > 0]
    # print(df)
    print(df['地下室面积'].describe())
    plot_scatter(df, '地下室面积', '销售价格')
