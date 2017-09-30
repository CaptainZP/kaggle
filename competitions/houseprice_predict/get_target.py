import pandas as pd


def get_target(file_path, col):
    df = pd.read_csv(file_path, usecols=col, encoding='gbk')
    return df

if __name__ == '__main__':
    file = "C:\\Users\\captainzp\\Desktop\\houseprice_train.csv"
    column = ['销售价格']
    target = get_target(file, column)
    target5 = target.head(5)
    print(target5)
