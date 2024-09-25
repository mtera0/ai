import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# lossの読み込み
train_acc_list = np.loadtxt("./train_cnn_train_acc_list.txt")
test_acc_list = np.loadtxt("./train_cnn_test_acc_list.txt")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()