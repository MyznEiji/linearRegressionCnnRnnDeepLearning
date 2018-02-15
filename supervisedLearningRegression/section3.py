# 必要なライブラリのimport
import numpy as np
import pandas as pd
import sklearn

# 図形描画
import matplotlib.pyplot as plt
import seaborn as sns


#データのインポート
from sklearn.datasets import load_boston
boston_data = sklearn.datasets.load_boston()

print(type(boston_data))

"""
Dictionary-like object, the interesting attributes are:
‘data’, the data to learn,
‘target’, the regression targets, and
‘DESCR’, the full description of the dataset.
"""
print("==================DESCR====================")
print(boston_data.DESCR)
print("==================data====================")
print(boston_data.data)
print("==================target====================")
print(boston_data.target)
print("==================feature_names====================")
print(boston_data.feature_names)

"""
X
- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    % lower status of the population

y
- MEDV     Median value of owner-occupied homes in $1000's
"""

"""
データをDataFrameオブジェクトに変換
pandas data frameの使い方

https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
"""

df_X = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
df_X
"""
X = (x1, x2, ...., x13)
"""

df_y = pd.DataFrame(data=boston_data.target, columns=["MEDV"])
df_y

"""
基礎統計情報を見てみる
"""

df_X.describe()

#相関係数を見てみる
df_X.corr()


#全変数の相関係数を見て、ヒートマップで可視化する。

#data frameを結合する
#横方向の時は、axis = 1を指定。縦方向の結合は、axisの指定なしでok
df_all = pd.concat([df_X, df_y], axis=1)

#相関係数を計算する
df_all.corr()

sns.heatmap(df_all.corr())
plt.show()

#高相関
#RM, LSTATとMEDVの散布図を描画する
sns.jointplot(x = "RM", y = "MEDV", data=df_all.loc[:,['RM','MEDV']])
plt.show()
sns.jointplot(x = "LSTAT", y = "MEDV", data=df_all.loc[:,['LSTAT','MEDV']])
plt.show()


#低相関
#DIS, とMEDVの散布図を描画する
sns.jointplot(x = "DIS", y = "MEDV", data=df_all.loc[:,['DIS','MEDV']])
plt.show()
sns.jointplot(x = "RAD", y = "MEDV", data=df_all.loc[:,['RAD','MEDV']])
plt.show()

"""
# Lesson3 sklearnによる線形回帰
"""


"""
## 訓練用データの準備　　

訓練データ作成
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""

#部屋の数vs家賃のデータセット
data_RM =df_all.loc[:,['RM','MEDV']]
data_RM

#訓練用データを作る
X = data_RM.RM.values.reshape((len(data_RM.RM), 1))
X
y = data_RM.MEDV
#訓練データとテストデータに分ける
#訓練:テスト = 3:1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(len(X_train))
print(len(X_test))

"""
## sklearnによるモデルの定義
参考

線形回帰
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""


from sklearn import linear_model
#線形回帰のためのオブジェクト

reg = linear_model.LinearRegression()


"""
## sklearnによる機械学習
"""

#訓練済みモデルの生成
#ここで、パラメータが最適化される
predictor = reg.fit(X_train, y_train)
print(predictor)

X_test

predictor.predict(X_test)
#テストデータでも正しいかplotしてみる
plt.scatter(X_test, y_test)
plt.plot(X_test, predictor.predict(X_test))
plt.show()


plt.scatter(y_test, predictor.predict(X_test))
plt.plot(y_test, y_test)
plt.show()


"""
# Lesson5 重回帰分析
## データの準備
"""


X = df_X.values
y = df_y.values

print(X)
print(y)

# 訓練データとテストデータに分ける
from sklearn.model_selection import train_test_split
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X, y, test_size=0.33)

## モデルの定義

#線形回帰のためのオブジェクト
reg_multi = linear_model.LinearRegression()

## 機械学習

#訓練済みモデルの生成
#ここで、パラメータが最適化される
predictor_multi = reg_multi.fit(X_train_multi, y_train_multi)

#データvs予測値でプロットしている
plt.scatter(y_test_multi, predictor_multi.predict(X_test_multi))
plt.show()
plt.plot(y_test_multi, y_test_multi)
plt.show()


## 精度の比較
from sklearn.metrics import mean_squared_error
#最小二乗誤差で比較する

print("linear : {}".format(mean_squared_error(y_test, predictor.predict(X_test))))
print("multi : {}".format(mean_squared_error(y_test_multi, predictor_multi.predict(X_test_multi))))

## 1次関数

def linear_function(X, a, b):
    return a*X + b

## 2次関数
def quadratic_function(X, a, b, c):
    return a*X**2 + b*X + c

## 3次関数
def cubic_function(X, a, b):
    pass a*X**3 + b*X**2 + c*X + d


# Lesson4 コスト関数の定義
## 二乗誤差損失関数の実装

## sklearnによる損失関数の定義

# Lesson5 機械学習の実装
## sklearnによる機械学習

## 勾配降下法の実装
