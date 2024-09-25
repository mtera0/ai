import sys
sys.path.append('..')
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from attention_seq2seq import AttentionSeq2seq


# データの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data(file_name_ja='mixed_1.3M_ja.txt',file_name_en='mixed_1.3M_en.txt',len_data=10000)
char_to_id_ja, id_to_char_ja, char_to_id_en, id_to_char_en = sequence.get_vocab_ja_en()
# 入力文を反転
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ハイパーパラメータの設定
vocab_size = max(len(char_to_id_ja)+1,len(char_to_id_en)+1)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 2
max_grad = 5.0

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

for _ in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

model.save_params()


# perplexityの描画（性能評価）
trainer.perplexity_plot()