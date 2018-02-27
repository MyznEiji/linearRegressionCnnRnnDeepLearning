"""
# ニューラルネットワークによる回帰と分類
# Lesson1 データの準備
"""

# 必要なライブラリのimport
import numpy as np
import pandas as pd
import sklearn

# 図形描画
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

print(mnist.COL_NAMES)
print("the number of data:{}".format(len(mnist.data)))
print(mnist.data[0].shape)
print(mnist.data[0])
plt.imshow(mnist.data[0].reshape(28,28), cmap='gray')
print(mnist.DESCR)
print("the number of target:{}".format(len(mnist.target)))
print(mnist.target)

plt.imshow(mnist.data[20000].reshape(28,28), cmap='gray')


#訓練データとテストデータの準備
from sklearn.model_selection import train_test_split
#X = np.array([e.astype(np.float32) for e in mnist.data])
#y = np.array([e.astype(np.int32) for e in mnist.target])
X = np.array(mnist.data).astype(np.float32)
y = np.array(mnist.target).astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(len(X_train), len(X_test))


"""
# Lesson2 Chainerによる多層パーセプトロンの実装
"""

import chainer
"""
## モデルを定義する
"""


import chainer.links as L
import chainer.functions as F

class MultiLayerNN(chainer.Chain):

    def __init__(self, hidden_layer=50, n_out=10):
        super(MultiLayerNN, self).__init__(
            l1=L.Linear(None, hidden_layer),
            l2=L.Linear(hidden_layer, hidden_layer),
            l3=L.Linear(hidden_layer, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

model = MultiLayerNN()


"""
## 最適化アルゴリズムを定義する
"""

from chainer import optimizers

# http://docs.chainer.org/en/stable/reference/optimizers.html

optimizer = optimizers.Adam()
optimizer.setup(model)

"""
## 学習イテレーションを実装する
"""


"""iterators.SerialIteratorの使い方
http://docs.chainer.org/en/stable/reference/generated/chainer.iterators.SerialIterator.html#chainer.iterators.SerialIterator

Chainerのdata setの形式
[(np.array([x,x,x, ... x, x], dtype=float32), y),
 (np.array([x,x,x, ... x, x], dtype=float32), y),
.
.
.
(np.array([x,x,x, ... x, x], dtype=float32), y)]
"""

from chainer import iterators

#バッチ学習のためのバッチサイズを指定
#とりあえず、訓練データ49000の1/100で指定
batch_size = 490

dataset_train = []

for X, y in zip(X_train, y_train):
    dataset_train.append((X, y))

train_iterator = iterators.SerialIterator(dataset_train, batch_size)

print(train_iterator.epoch)

for _ in range(100):
    #next()でバッチのデータを返す
    len(train_iterator.next())

print(train_iterator.epoch)


for _ in range(1000):
    train_iterator.next()

print(train_iterator.epoch)


import numpy as np

max_epoch = 10
train_acc_log = []
test_acc_log = []
train_loss_log = []
test_loss_log = []

train_iterator = iterators.SerialIterator(dataset_train, batch_size)

while train_iterator.epoch < max_epoch:
    #バッチのデータを用意
    batch = train_iterator.next()
    X_batch, y_batch = chainer.dataset.concat_examples(batch)

    #コストを計算
    y_train_predicted = model(X_batch)
    loss_train = F.softmax_cross_entropy(y_train_predicted, y_batch)

    #学習
    model.cleargrads()
    loss_train.backward()
    optimizer.update()

    #精度の確認
    acc_train = F.accuracy(y_train_predicted, y_batch)
    train_acc_log.append(float(acc_train.data))
    train_loss_log.append(float(loss_train.data))



    #test
    if train_iterator.is_new_epoch:
        print("===============================================")
        y_test_predicted = model(X_test)
        loss_test = F.softmax_cross_entropy(y_test_predicted, y_test)
        acc_test = F.accuracy(y_test_predicted, y_test)

        test_acc_log.append(float(acc_test.data))
        test_loss_log.append(float(loss_test.data))

        print("epoch : {}".format(train_iterator.epoch))
        print("train loss : {}".format(float(loss_train.data)))
        print("train acc : {}".format(float(acc_train.data)))
        print("test loss : {}".format(float(loss_test.data)))
        print("test acc : {}".format(float(acc_test.data)))



plt.plot(range(len(train_acc_log)),train_acc_log)
plt.plot(range(100, len(test_acc_log)*101, 100), test_acc_log)
plt.show()


"""
# tensorflowの使い方
インストール方法
https://www.tensorflow.org/install/
"""

import tensorflow as tf

"""
## basic concept

1. グラフ(computational graph)の構築
2. グラフ(computational graph)の実行

### グラフとは???

tensorflowでは、ノードというものを定義している。

何かinputがあって、何かoutputするもの。

inputがNoneの場合、定数(constant)として扱う
"""

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(2.0, dtype=tf.float32)
print(node1, node2)

sess = tf.Session()
sess.run([node1, node2])


node3 = tf.add(node1, node2)

sess.run(node3)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = tf.add(a,b)

sess.run(adder_node, {a:4, b:2})

sess.run(adder_node, {a:[1,2], b:[3,4]})

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W*x + b

init = tf.global_variables_initializer()
sess.run(init)


sess.run(linear_model, {x:[1,2,3,4]})


y = tf.placeholder(tf.float32)
square_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(square_delta)
sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)

for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
sess.run([W, b])


"""
# tensorflowによる多層パーセプトロンの実装
## グラフをつないでいくような感覚で全結合ニューラルネットワークを書いてみよう

参考:
https://github.com/pinae/TensorFlow-MNIST-example/blob/master/fully-connected.py

## 必要なデータ

- 28*28の画像を784の長さの配列として
- 教師データとして、one hot vectorを作る
"""


from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")

X = mnist.data
y = mnist.target

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(range(0,10))
y_onehot = lb.transform(y)

x_input = tf.placeholder(tf.float32, [None, 784])
y_teacher = tf.placeholder(tf.float32, [None, 10])

keep_prob_input = tf.placeholder(tf.float32)

###第一層###
x_input_layer = tf.nn.dropout(x_input, keep_prob=keep_prob_input)
#50

def weight_variable(shape):
    #標準偏差が0.01の切断正規分布にしたがって初期値をランダム生成
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

hidden_layer_size = 50

W_fc1 = weight_variable([784, hidden_layer_size])
b_fc1 = bias_variable([hidden_layer_size])

#relu関数を活性化関数として使用
h_fc1 = tf.nn.relu(tf.matmul(x_input_layer, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)

#ドロップアウト済みの出力
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



###第二層###
W_fc2 = weight_variable([hidden_layer_size, hidden_layer_size])
b_fc2 = bias_variable([hidden_layer_size])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

###第三層###
#出力は0~9までの10個を、softmax関数で確率として出力
W_fc3 = weight_variable([hidden_layer_size, 10])
b_fc3 = bias_variable([10])
y_out = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

#コスト関数
cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y_teacher*tf.log(y_out), reduction_indices=[1]))

#最適化関数
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)

#iteration
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_teacher, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = map(lambda x : np.array(x).astype(np.float32), train_test_split(X, y_onehot, test_size=0.3))

#train:49000
#test :2100
import random
saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver.save(sess, "mnist_fc_best")

epoch_size = 20
batch_size = 100
best_accuracy = 0.0

def random_sample(X, y, size=100):
    idx = range(0, len(y))
    random_idx = random.sample(idx, size)
    return X[random_idx, :], y[random_idx, :]


for epoch in range(1,epoch_size+1):

    for i in range(int(len(y_train)/batch_size)):
        X_batch, y_batch = random_sample(X_train, y_train, batch_size)
        if i == 0:
            print("=====================")
            train_accuracy = sess.run(accuracy, feed_dict = {
                    x_input:X_batch, y_teacher:y_batch, keep_prob_input:1.0, keep_prob:1.0
            })
            print("{} : training accuracy {}%".format(epoch, train_accuracy*100))
            test_accuracy = sess.run(accuracy, feed_dict={
                x_input: X_test, y_teacher: y_test, keep_prob_input: 1.0, keep_prob: 1.0})
            print("{} : test accuracy {}%".format(epoch, test_accuracy*100))

            #ループの中で、最高の精度を持つネットワークを保存したい
            if test_accuracy >= best_accuracy:
                saver.save(sess, 'mnist_fc_best')
                best_accuracy = test_accuracy
                print("Validation accuracy improved: {}%. Saving the network.".format(test_accuracy*100))
            else:
                #テストaccuracyが下がったときは、過学習なので1epoch前のモデルに戻す
                saver.restore(sess, 'mnist_fc_best')
                print("restore!!!! now : {}, before : {}".format(test_accuracy*100, best_accuracy*100))

        sess.run(optimizer, feed_dict={
                x_input: X_batch, y_teacher: y_batch, keep_prob_input: 0.9, keep_prob: 1.0})


"""
# chainerによるCNNの実装

参考1:http://docs.chainer.org/en/stable/tutorial/convnet.html


参考2:http://yann.lecun.com/exdb/lenet/


## モデルの定義
参考1 :http://docs.chainer.org/en/stable/reference/links.html


参考2 :http://docs.chainer.org/en/stable/reference/functions.html
"""


import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers

class LeNet5(chainer.Chain):

    def __init__(self, hidden_layer = 84, n_out = 10):
        super(LeNet5, self).__init__(
            conv1 = L.Convolution2D(in_channels=None, out_channels=6, ksize=5, stride=1),
            conv2 = L.Convolution2D(in_channels=None, out_channels=16, ksize=5, stride=1),
            conv3 = L.Convolution2D(in_channels=None, out_channels=120, ksize=4, stride=1),
            fc4 = L.Linear(None, hidden_layer),
            fc5 = L.Linear(hidden_layer, n_out),
        )

    def __call__(self, x):
        h = F.sigmoid(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv2(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv3(x))
        h = F.sigmoid(self.fc4(h))
        return F.sigmoid(self.fc5(h))

model = LeNet5()

optimizer = optimizers.Adam()
optimizer.setup(model)


from chainer import iterators
#訓練データとテストデータの準備
from sklearn.model_selection import train_test_split

X = np.array(mnist.data).astype(np.float32)
y = np.array(mnist.target).astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#(len(X_train), 784)
#(len(X_train), channel, height, width)

X_train = X_train.reshape((-1, 1, 28,28))
X_test = X_test.reshape((-1, 1, 28,28))

#バッチ学習のためのバッチサイズを指定
#とりあえず、訓練データ49000の1/100で指定
batch_size = 490

dataset_train = []

for X, y in zip(X_train, y_train):
    dataset_train.append((X, y))

train_iterator = iterators.SerialIterator(dataset_train, batch_size)

import numpy as np

max_epoch = 10
train_acc_log = []
test_acc_log = []
train_loss_log = []
test_loss_log = []

train_iterator = iterators.SerialIterator(dataset_train, batch_size)

while train_iterator.epoch < max_epoch:
    #バッチのデータを用意
    print("!!!!!!!!!!!!!!")
    batch = train_iterator.next()
    X_batch, y_batch = chainer.dataset.concat_examples(batch)

    #コストを計算
    y_train_predicted = model(X_batch)
    loss_train = F.softmax_cross_entropy(y_train_predicted, y_batch)

    #学習
    model.cleargrads()
    loss_train.backward()
    optimizer.update()

    #精度の確認
    acc_train = F.accuracy(y_train_predicted, y_batch)
    train_acc_log.append(float(acc_train.data))
    train_loss_log.append(float(loss_train.data))

    #test
    if train_iterator.is_new_epoch:
        print("===============================================")
        y_test_predicted = model(X_test)
        loss_test = F.softmax_cross_entropy(y_test_predicted, y_test)
        acc_test = F.accuracy(y_test_predicted, y_test)

        test_acc_log.append(float(acc_test.data))
        test_loss_log.append(float(loss_test.data))

        print("epoch : {}".format(train_iterator.epoch))
        print("train loss : {}".format(float(loss_train.data)))
        print("train acc : {}".format(float(acc_train.data)))
        print("test loss : {}".format(float(loss_test.data)))
        print("test acc : {}".format(float(acc_test.data)))


"""
# tensorflowによるCNNの実装
参考1 :https://www.tensorflow.org/tutorials/deep_cnn

参考2 :https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py#
"""

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X = mnist.data
y = mnist.target

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(range(0,10))
print(lb.classes_)
y_onehot = lb.transform(y)
print(y_onehot[0], y[0])

X_train, X_test, y_train, y_test = map(lambda x : np.array(x).astype(np.float32), train_test_split(X, y_onehot, test_size=0.3))


def conv2d(x, W):
    #[1,stride, stride, 1]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def weight_variable(shape):
    #標準偏差0.01で切断正規分布(truncated normal distribution)にしたがって初期値をランダム生成
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def lenet5(x):
    x_input = tf.reshape(x, [-1, 28 ,28,1])

    #畳み込み1
    W_conv1 = weight_variable([5,5,1,6])
    b_conv1 = bias_variable([6])
    h_conv1 = tf.nn.sigmoid(conv2d(x_input, W_conv1)+b_conv1)
    #14*14
    h_pool1 = max_pool_2x2(h_conv1)

    #畳み込み2
    W_conv2 = weight_variable([5,5,6,16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2)+b_conv2)
    #7*7
    h_pool2 = max_pool_2x2(h_conv2)

    #畳み込み3
    W_conv3 = weight_variable([4,4,16,120])
    b_conv3 = bias_variable([120])
    h_conv3 = tf.nn.sigmoid(conv2d(h_pool2, W_conv3)+b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*120])

    #全結合1
    W_fc1 = weight_variable([7*7*120, 84])
    b_fc1 = bias_variable([84])

    h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    #全結合2
    W_fc2 = weight_variable([84, 10])
    b_fc2 = bias_variable([10])
    h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

    return h_fc2


    x_input = tf.placeholder(tf.float32, [None, 784])
y_teacher = tf.placeholder(tf.float32, [None, 10])

y_out = lenet5(x_input)

#コスト関数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_teacher,logits=y_out)
cross_entropy = tf.reduce_mean(cross_entropy)

#最適化関数
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)

#ソフトマックスの確率最大の場所が、
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_teacher, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#train:49000なので、100バッチ490回
import random
# create a saver
saver = tf.train.Saver()

# initialize the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver.save(sess, 'mnist_fc_best')

batch_size = 100
epoch_size = 20
best_accuracy = 0.0

def random_sample(X, y, size = 100):
    idx = range(0 , len(y))
    random_idx = random.sample(idx, size)
    return X[random_idx, :], y[random_idx, :]

for epoch in range(1, epoch_size+1):
    #バッチ学習
    for i in range(int(len(y_train)/batch_size)):
        X_batch, y_batch = random_sample(X_train, y_train, 100)
        #学習
        #バッチの初めにプリント
        if i == 0:
            print("=======================================")
            #精度確認のときは、drop outしない
            train_accuracy = sess.run(accuracy, feed_dict={
                x_input: X_batch, y_teacher : y_batch})
            print("{} : training accuracy {}%".format(epoch, train_accuracy*100))
            test_accuracy = sess.run(accuracy, feed_dict={
                x_input: X_test, y_teacher: y_test})
            print("{} : test accuracy {}%".format(epoch, test_accuracy*100))

            #ループの中で、最高の精度を持つネットワークを保存したい
            if test_accuracy >= best_accuracy:
                saver.save(sess, 'mnist_fc_best')
                best_accuracy = test_accuracy
                print("Validation accuracy improved: {}%. Saving the network.".format(test_accuracy*100))
            else:
                #テストaccuracyが下がったときは、過学習なので1epoch前のモデルに戻す
                saver.restore(sess, 'mnist_fc_best')
                print("restore!!!! now : {}, before : {}".format(test_accuracy*100, best_accuracy*100))

        #バッチ学習
        #過学習を防ぐためにdrop out率を設定
        sess.run(optimizer, feed_dict={
                x_input: X_batch, y_teacher: y_batch})

print("Best test accuracy: %g" % best_accuracy*100)


"""
# chainerによるRNNの実装
参考:http://docs.chainer.org/en/stable/tutorial/recurrentnet.html
"""

"""
# tensorflowによるRNNの実装

参考:https://www.tensorflow.org/tutorials/recurrent
"""
