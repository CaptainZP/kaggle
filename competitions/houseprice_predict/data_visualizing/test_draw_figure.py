import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
X = np.linspace(-np.pi, np.pi, 20, endpoint=True)
(C, S) = np.cos(X), np.sin(X)
T = np.linspace(-1, 1, 20)
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)
print(len(C), len(S), len(T))

# subplots 可绘制多子图
# fig, axes = plt.subplots(1, 2)
# # --------------------------------------
# axes[0].scatter(X, C, c='r', label='cos')   # x轴数组，y轴数组，颜色线型，标签
# axes[0].legend(loc='upper right')   # 标签的位置
# axes[0].set_xlabel('x轴', rotation='horizontal')
# axes[0].set_ylabel('y轴', rotation='horizontal')
# axes[0].set_xlim(X.min()*1.5, X.max()*1.5)   # 调整轴的范围
# axes[0].set_ylim(C.min()*2, C.max()*2)
# axes[0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])   # 调整轴的刻度
# axes[0].set_yticks([-1, 0, 1])
# # ----------------------------------------
# axes[1].plot(X, S, 'b-', label='sin')
# axes[1].legend(loc='upper right')   # 标签的位置
# axes[1].set_xlabel('x轴', rotation='horizontal')
# axes[1].set_ylabel('y轴', rotation='horizontal')
# axes[1].set_xlim(X.min()*1.1, X.max()*1.1)   # 调整轴的范围
# axes[1].set_ylim(C.min()*1.1, C.max()*1.1)
# axes[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])   # 调整轴的刻度
# axes[1].set_yticks([-1, 0, 1])
# # -------------------------------------------
# fig.subplots_adjust(hspace=0.4)
# plt.show()   # 显示图

# figure
plt.figure(figsize=(10, 6))
# plt.scatter(X, C, c='r', label='cos')   # 散点图， x轴数组，y轴数组，颜色，标签
# plt.plot(X, S, 'b-', label='sin')   # 折线图， x轴数组，y轴数组，颜色线型，标签
# plt.bar(X, T, color='y', label='bar', width=0.1)
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='pink', edgecolor='black', label='hist')  # bins是分成几个柱子
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, '--')
plt.legend(loc='upper right')   # 标签的位 置
plt.xlabel('x轴', rotation='horizontal')
plt.ylabel('y轴', rotation='horizontal')

# plt.xlim(X.min()*1.1, X.max()*1.1)   # 调整轴的范围
# plt.ylim(C.min()*1.1, C.max()*1.1)
# plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])   # 调整轴的刻度
# plt.yticks([-1, 0, 1])

# plt.savefig('result.png')
plt.show()   # 显示图