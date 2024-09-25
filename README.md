# ai
NNを基礎とするAIについて理解するために作成。

## cnn
### 1.プロジェクト概要
datasetとしてMNISTを利用したNNとCNNの精度比較。

### 2.主要技術

| 言語・モジュール | バージョン |
| -------------------- | ---------- |
| Python                | 3.9.12       |
| matplotlib                | 3.6.2       |
| numpy                | 1.21.6       |
| pandas                | 1.5.3       |
| scikit-learn                | 1.5.1       |
| scipy                | 1.7.3       |

### 3.開発環境構築方法

#### 1. Python のインストール

Python をインストール。インストール方法は[公式ドキュメント](https://www.python.org/downloads/)を参照。

#### 2. pandas, spotipy, python-dotenvのインストール

[matplotlib](https://matplotlib.org/stable/users/getting_started/), [numpy](https://numpy.org/ja/install/), [scikit-learn](https://scikit-learn.org/stable/install.html),[scipy](https://scipy.org/install/),  のうちインストールしていないものがあれば各リンク先のコマンドによりインストール

#### 4.性能

作成したモデルの性能は下図のようになった。

<img src="https://github.com/user-attachments/assets/e48abb03-8e47-4cab-a51f-afcfcb29289b" width="400"><img src="https://github.com/user-attachments/assets/75b18731-97eb-4675-8b44-972502eb536b" width="400">

横軸はepoch数、縦軸はaccuracyであり、左がNN、右がCNNの実行結果となっている。
epochが大きくなるに従っていずれもaccuracyが大きくなっているが、NNでは0.8付近、CNNでは1.0付近に収束している。CNNでは畳み込みやPoolingなどを利用しているため性能が改善されていることがわかる。



## nlp
### 1.プロジェクト概要
datasetとしてMNISTを利用したNNとCNNの精度比較。

### 2.主要技術

| 言語・モジュール | バージョン |
| -------------------- | ---------- |
| Python                | 3.9.12       |
| matplotlib                | 3.6.2       |
| numpy                | 1.21.6       |

### 3.開発環境構築方法

#### 1. Python のインストール

Python をインストール。インストール方法は[公式ドキュメント](https://www.python.org/downloads/)を参照。

#### 2. pandas, spotipy, python-dotenvのインストール

[matplotlib](https://matplotlib.org/stable/users/getting_started/), [numpy](https://numpy.org/ja/install/)  のうちインストールしていないものがあれば各リンク先のコマンドによりインストール

#### 4.性能

作成したモデルの性能は下図のようになった。
<div align="center">
  <img src="https://github.com/user-attachments/assets/42c444fd-7984-4fec-994c-40c259dcd9c4">
</div>
横軸はepoch数、縦軸はperplexityという言語モデルにおける精度の良さを表す指標である。予測される選択肢の数がどれくらい絞られているかを表しており小さいほど精度が良いとされる。
学習が進むにつれてperplexityの大きさは10程度まで減少していることから、モデルの予測精度が高まっていることが確認できる。
