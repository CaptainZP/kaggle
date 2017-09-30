import numpy as np

a = np.array([2, 3, 1, 5])
b = np.array([4, 6, 2, 10])
index = np.argsort(a, axis=0)
print(a[index])
print(b[index])

