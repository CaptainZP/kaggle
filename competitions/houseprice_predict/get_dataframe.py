import pandas as pd


def get_dataframe(file_path, cols=None):
    '''
    得到想要的列的DataFrame
    :param file_path：csv文件路径，str
    :param cols: 想要读取的列，str list
    :param date_col: 日期所在的列，将其转换成日期类型，str
    :return: DataFrame
    '''
    df = pd.read_csv(file_path, usecols=cols, encoding='gbk')
    return df


if __name__ == '__main__':
    file = "C:\\Users\\captainzp\\Desktop\\houseprice_train.csv"
    columns = ['销售日期', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数', '房屋评分', '建筑面积',
               '地下室面积', '建筑年份', '修复年份', '纬度', '经度']
    features = get_dataframe(file, columns)
    print(features.head(10))
