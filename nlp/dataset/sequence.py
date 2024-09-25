import sys
sys.path.append('..')
import os
import numpy

from janome.tokenizer import Tokenizer

id_to_char = {}
char_to_id = {}

id_to_char_ja = {}
char_to_id_ja = {}

id_to_char_en = {}
char_to_id_en = {}

def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char

def _update_vocab_ja(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id_ja:
            tmp_id = len(char_to_id_ja)
            char_to_id_ja[char] = tmp_id
            id_to_char_ja[tmp_id] = char

def _update_vocab_en(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id_en:
            tmp_id = len(char_to_id_en)
            char_to_id_en[char] = tmp_id
            id_to_char_en[tmp_id] = char



def load_data(file_name_ja='',file_name_en='', seed=1984,len_data=-1):
    file_path_ja = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name_ja
    file_path_en = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name_en

    
    if not os.path.exists(file_path_ja):
        print('No file for Japanese: %s' % file_name_ja)
        return None
    if not os.path.exists(file_path_en):
        print('No file for English: %s' % file_name_en)
        return None



    sentence_japanese, sentence_english = [], []

    for line in open(file_path_ja, 'r'):
        sentence_japanese.append(line)
    for line in open(file_path_en, 'r'):
        sentence_english.append(line)
    
    if len_data != -1:
        sentence_japanese = sentence_japanese[:len_data]
        sentence_english = sentence_english[:len_data]

    # create vocab dict
    t = Tokenizer(wakati=True)
    for i in range(len(sentence_japanese)):
        ja, en = sentence_japanese[i], sentence_english[i]
        split_ja = "starttoken " + " ".join(t.tokenize(ja)) + " >"
        split_en = "< " + en.replace(".", " .").replace("?", " ?").replace(",", " ,").replace("\n", "") + " endtoken"
        _update_vocab_ja(split_ja.split(" "))
        _update_vocab_en(split_en.split(" "))
        sentence_japanese[i] = split_ja.split(" ")
        sentence_english[i] = split_en.split(" ")
    


    # create numpy array
    len_sentence_ja_max = max([len(x) for x in sentence_japanese])
    len_sentence_en_max = max([len(x) for x in sentence_english])
    x = numpy.zeros((len(sentence_japanese), len_sentence_ja_max), dtype=int)
    t = numpy.zeros((len(sentence_japanese), len_sentence_en_max), dtype=int)


    for i, sentence in enumerate(sentence_japanese):
        if len(sentence) < len_sentence_ja_max:
            sentence += ['>' for _ in range(len_sentence_ja_max - len(sentence))]
        x[i] = [char_to_id_ja[c] for c in list(sentence)]
    for i, sentence in enumerate(sentence_english):
        if len(sentence) < len_sentence_en_max:
            sentence += ['endtoken' for _ in range(len_sentence_en_max - len(sentence))]
        t[i] = [char_to_id_en[c] for c in list(sentence)]



    # shuffle
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab():
    return char_to_id, id_to_char

def get_vocab_ja_en():
    return char_to_id_ja, id_to_char_ja, char_to_id_en, id_to_char_en