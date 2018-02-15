"""
# Lesson1 データの準備
## ライブラリをインポートする

"""

# 必要なライブラリのimport
import numpy as np
import pandas as pd
import sklearn

# 図形描画
import matplotlib.pyplot as plt
import seaborn as sns

"""
## データを見る
"""


from sklearn import datasets
# import some data to play with
iris = datasets.load_iris()

print("==================DESCR====================")
print(iris.DESCR)
print("==================feature_names====================")
print(iris.feature_names)
print("==================data====================")
print(iris.data)
print("==================target====================")
print(iris.target)
print("==================target_names====================")
print(iris.target_names)


"""
# Lesson2 pandasとmatplotlibでデータを可視化
## データをDataFrameオブジェクトに変換

pandas data frameの使い方

https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
"""


df_X = pd.DataFrame(iris.data, columns = iris.feature_names)

df_y = pd.DataFrame(iris.target, columns=["iris_name"])

df_all = pd.concat([df_X, df_y], axis=1)
df_all

## 基礎統計情報を見てみる

print(df_all.describe())
sns.heatmap(df_all.corr())

sns.pairplot(df_all)


def describe_scatter(df_X, df_y, i, j):

    plt.figure(2, figsize=(8,6))

    X1, X2 = df_X[df_X.columns[i]], df_X[df_X.columns[j]]
    #https://matplotlib.org/users/colormaps.html
    plt.scatter(X1, X2, c=df_y, cmap=plt.cm.Accent)

    plt.xlabel(df_X.columns[i])
    plt.ylabel(df_X.columns[j])

    #軸の設定
    #プロットのために範囲を取得
    x1_min, x1_max = X1.min() - 0.5, X1.max() + 0.5
    x2_min, x2_max = X2.min() - 0.5, X2.max() + 0.5

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xticks(())
    plt.yticks(())

describe_scatter(df_X, df_y, 0, 1)

describe_scatter(df_X, df_y, 0,2)

"""
# Lesson3 sklearnによるロジスティック回帰
## 訓練用データの準備　　

訓練データ作成
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""

from sklearn import linear_model

X = df_X.values[:, :2]
y = df_y.values.reshape(-1)

#logistic regression用インスタンス
log_reg = linear_model.LogisticRegression()

log_reg.fit(X,y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))


print(np.array([xx.reshape(-1), yy.reshape(-1)]).T)

result_all_mesh = log_reg.predict(np.array([xx.reshape(-1), yy.reshape(-1)]).T)

result_all_mesh = result_all_mesh.reshape(xx.shape)

plt.figure(1, figsize=(8, 6))
plt.pcolormesh(xx, yy, result_all_mesh, cmap=plt.cm.tab10)

#精度確認のために、教師データをプロット
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.tab10)

"""
# Lesson4 SVM
## モデルの定義と学習

"""


from sklearn.svm import SVC
svc = SVC()
svc.fit(X, y)
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

def describe_scatter_cgrid(model, X, y):
    model.fit(X,y)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))


    print(np.array([xx.reshape(-1), yy.reshape(-1)]).T)

    result_all_mesh = model.predict(np.array([xx.reshape(-1), yy.reshape(-1)]).T)

    result_all_mesh = result_all_mesh.reshape(xx.shape)

    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx, yy, result_all_mesh, cmap=plt.cm.tab10)

    #精度確認のために、教師データをプロット
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.tab10)
    plt.show()

describe_scatter_cgrid(SVC(), X, y)

"""
## 過学習の過程
"""

describe_scatter_cgrid(SVC(C=10), X, y)

describe_scatter_cgrid(SVC(C=1000), X, y)

describe_scatter_cgrid(SVC(C=10), X, y)

"""
# Lesson5 ニューラルネットワークを用いた分類
## モデルの定義と学習
"""


#multi layer perseptron
from sklearn.neural_network import MLPClassifier

#http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html


nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=10, random_state = 5)

describe_scatter_cgrid(nn_clf, X, y)


"""
## grid searchとcross validation
"""

from sklearn.model_selection import GridSearchCV

#探索したいパラメータ
tuned_parameters = {"solver":["lbfgs"], "alpha":[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                     "hidden_layer_sizes":list(range(3,30,3))}

#グリッドサーチ
grid_cv = GridSearchCV(MLPClassifier(random_state=5), tuned_parameters, cv=5)

describe_scatter_cgrid(grid_cv, X, y)

grid_cv.best_params_
