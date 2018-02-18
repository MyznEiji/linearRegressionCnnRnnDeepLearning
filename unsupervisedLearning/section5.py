"""
# 教師なし学習
# データの準備
## ライブラリのインポート
"""

# 必要なライブラリのimport
import numpy as np
import pandas as pd
import sklearn

# 図形描画
import matplotlib.pyplot as plt
import seaborn as sns

import chainer
import tensorflow

print(sklearn.__version__)
print(np.__version__)
print(pd.__version__)
print(chainer.__version__)
print(tensorflow.__version__)

"""
## データを見る
"""

# for PCA
from sklearn.datasets import load_wine
wine_data = sklearn.datasets.load_wine()

"""
Dictionary-like object, the interesting attributes are:
‘data’, the data to learn,
‘target’, the regression targets, and
‘DESCR’, the full description of the dataset.
"""
print("==================DESCR====================")
print(wine_data.DESCR)
print("==================data====================")
print(wine_data.data)
print("==================target====================")
print(wine_data.target)
print("==================feature_names====================")
print(wine_data.feature_names)

"""
# Lesson2 pandasとmatplotlibを使ったデータの理解
"""

# データをDataFrameオブジェクトに変換

"""
pandas data frameの使い方

https: // pandas.pydata.org / pandas - docs / stable / generated / pandas.DataFrame.html
"""

df_X = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df_y = pd.DataFrame(data=wine_data.target, columns=["wine class"])

df_X

# 前回定義した可視化関数


def describe_scatter(X, y, i, j):
    plt.figure(2, figsize=(8, 6))

    # プロット
    # カラーを選択
    X1, X2 = X[X.columns[i]], X[X.columns[j]]
    plt.scatter(X1, X2, c=y, cmap=plt.cm.tab10)
    plt.xlabel(X.columns[i])
    plt.ylabel(X.columns[j])

    # 軸の設定
    # プロットのために範囲を取得
    x1_min, x1_max = X1.min() - 0.5, X1.max() + 0.5
    x2_min, x2_max = X2.min() - 0.5, X2.max() + 0.5

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xticks(())
    plt.yticks(())

# 全変数の相関係数を見て、ヒートマップで可視化する。


# data frameを結合する
# 横方向の時は、axis = 1を指定。縦方向の結合は、axisの指定なしでok
df_all = pd.concat([df_X, df_y], axis=1)

# 相関係数を計算する
df_all.corr()

sns.heatmap(df_all.corr())

sns.pairplot(df_X)

describe_scatter(df_X, df_y, 3, 5)

"""
# Lesson3 主成分分析(Principal component analysis)
## sklearnによるPCA
参考

線形回帰
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""

from sklearn.decomposition import PCA

# モデルの生成
# 相関が高い3項目に対して、2次元まで次元を下げてみる
X_reduced = PCA(n_components=2).fit_transform(
    df_X.iloc[:, [3, 5, 6, 7, 11]].values)

plt.figure(2, figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df_y, cmap=plt.cm.tab10)
plt.show()

"""
# Lesson4 k-means法によるクラスタリング

#参考
#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#example-cluster-plot-kmeans-silhouette-analysis-py%5D

## ダミーデータの生成
"""

# for k-means
from sklearn.datasets import make_blobs
# 参考:http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html


X_kmeans, y_kmeans = make_blobs(n_samples=300,
                                n_features=2,
                                cluster_std=1,
                                centers=3,
                                shuffle=True,
                                random_state=3)
print(X_kmeans)
print("================")
print(y_kmeans)

df_X_kmeans = pd.DataFrame(X_kmeans)
df_y_kmeans = pd.DataFrame(y_kmeans)

describe_scatter(df_X_kmeans, df_y_kmeans, 0, 1)

"""
## クラス数を指定したk-means クラスタリング
"""

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3).fit(X_kmeans)
describe_scatter(df_X_kmeans, pd.DataFrame(kmeans.labels_), 0, 1)


"""
最適なクラス数がわからない時のシルエット分析
"""
