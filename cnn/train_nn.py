from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

from nn import TwoLayerNet

mnist = fetch_openml('mnist_784')

train_size = 5000 # 訓練データ
test_size = 1000 # テストデータ

# データの読み込み
x_train, x_test, t_train, t_test = train_test_split(
        mnist.data, mnist.target, # 分割対象のデータ
        stratify=mnist.target, # 指定すると分割後にデータの割合が均等となるように分割される
        random_state=66, # データがシャッフルされる時に利用されるシード値
        train_size=train_size, test_size=test_size # 訓練用とテスト用で取り出すデータ数を指定
        )

x_train, x_test, t_train, t_test = x_train.to_numpy(), x_test.to_numpy(), np.array(t_train.to_numpy(),dtype='int64'), np.array(t_test.to_numpy(),dtype='int64')
t_train_one_hot = np.identity(10, dtype='int64')[t_train]
t_test_one_hot = np.identity(10, dtype='int64')[t_test]

#print("shape x_train : ",x_train.shape)
#print("0 th element of x_train : ",x_train[0])
#print("type x_train : ",type(x_train))
#x_train_size = x_train.shape[0]

# ハイパーパラメータ
iters_num = 3000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 1epochあたりの繰り返し回数
iter_per_epoch = max(train_size/batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size) # 0以上train_size未満の整数配列からbatch_size個を抽出
    x_batch = x_train[batch_mask]
    t_batch = t_train_one_hot[batch_mask]

#    print("x_batch : ",x_batch)
#    print("t_batch : ",t_batch)
    # 勾配の計算
    #grad = network.numerical_gradient(x_batch,t_batch)
    grad = network.gradient(x_batch,t_batch)

    # パラメータ更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1epoch ごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train_one_hot)
        test_acc = network.accuracy(x_test, t_test_one_hot)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | "+str(train_acc)+", "+str(test_acc))

np.savetxt("./train_nn_train_acc_list.txt", train_acc_list)
np.savetxt("./train_nn_test_acc_list.txt", test_acc_list)