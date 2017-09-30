import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve


def plot_validation_curve(estimator, X, y, param_name, param_range, title, xlabel_name):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=10, scoring="r2", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print('train_scores_mean:', train_scores_mean)
    print('train_scores_std:', train_scores_std)
    print('test_scores_mean:', test_scores_mean)
    print('test_scores_std:', test_scores_std)


    plt.title(title)
    plt.xlabel(xlabel_name)
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)   # 纵坐标范围
    lw = 2   # 线宽

    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    digits = load_digits()
    X, y = digits.data, digits.target
    estimator = SVC()
    para_name = 'gamma'
    para_range = np.logspace(-6, -1, 5)
    title = 'SVC'
    xlabel_name = '$\gamma$'
    plot_validation_curve(estimator, X, y, para_name, para_range, title, xlabel_name)
