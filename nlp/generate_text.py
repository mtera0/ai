import sys
sys.path.append('..')
import numpy as np
from attention_seq2seq import AttentionSeq2seq
from dataset import sequence

from janome.tokenizer import Tokenizer

(x_train, t_train), (x_test, t_test) = sequence.load_data(file_name_ja='mixed_1.3M_ja.txt',file_name_en='mixed_1.3M_en.txt',len_data=10000)
char_to_id_ja, id_to_char_ja, char_to_id_en, id_to_char_en = sequence.get_vocab_ja_en()

len_sentence_ja_max = 145
len_sentence_en_max = 90

# start文字とskip文字の設定
input_txt = "私は猫である。"
t = Tokenizer(wakati=True)
ja = "starttoken " + " ".join(t.tokenize(input_txt)) + " >"
split_ja = ja.split(" ")
if len(split_ja) < len_sentence_ja_max:
    split_ja += ['>' for _ in range(len_sentence_ja_max - len(split_ja))]
split_en = "<"
start_id = char_to_id_en[split_en]
x = np.zeros((1,len_sentence_ja_max),dtype=int)
x[0] = [char_to_id_ja[c] for c in list(split_ja)]


vocab_size = max(len(char_to_id_ja)+1,len(char_to_id_en)+1)
wordvec_size = 16
hidden_size = 256

model = AttentionSeq2seq(vocab_size,wordvec_size,hidden_size)
model.load_params('AttentionSeq2seq.pkl')



# 文章生成
word_ids = model.generate(xs=x,start_id=start_id,sample_size=len_sentence_en_max)
txt = ' '.join([id_to_char_en[i] for i in word_ids])
txt = txt.replace(' endtoken', '.\n').replace('endtoken','')

print(txt)
