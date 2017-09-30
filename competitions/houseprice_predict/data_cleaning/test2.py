import pandas as pd


df = pd.DataFrame({'a': [11, 22, 33, 44], 'b': [1, 0, 3, 0], 'c': [10, 20, 30, 40]})
print(df)
df['d'] = df['a'] - df['b']*10
print(df)
index1 = df[df['b'] == 0].index
print(index1)
print(df['d'][index1])
df['d'][index1] = df['a'][index1] - df['c'][index1]
print(df['d'])
