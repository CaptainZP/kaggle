from houseprice_predict.get_dataframe import get_dataframe


def var_dtype_count(df):
    '''
    :param df: dataframe
    :return: 各种变量个数
    '''
    quantitative = [v for v in df.columns if df.dtypes[v] != 'object']  # 数值型
    qualitative = [v for v in df.columns if df.dtypes[v] == 'object']  # 类别型
    return len(quantitative), len(qualitative)

if __name__ == '__main__':
    file = "C:\\Users\\captainzp\\Desktop\\kc_train.csv"
    columns = ['销售日期', '销售价格', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数', '房屋评分',
                       '建筑面积', '地下室面积', '建筑年份', '修复年份', '纬度', '经度']
    df = get_dataframe(file, columns)
    df['房屋评分'] = df['房屋评分'].astype(str)
    quantitative_len, qualitative_len = var_dtype_count(df)
    print("数值变量个数: {}, 类型变量个数: {}".format(quantitative_len, qualitative_len))
