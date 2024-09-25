from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

from cnn import SimpleConvNet
from common.trainer import Trainer

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

x_train = x_train.reshape(-1,1,28,28)
x_test = x_test.reshape(-1,1,28,28)

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train_one_hot, x_test, t_test_one_hot,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

np.savetxt("./train_cnn_train_acc_list.txt", trainer.train_acc_list)
np.savetxt("./train_cnn_test_acc_list.txt", trainer.test_acc_list)

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")