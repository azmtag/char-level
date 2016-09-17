# coding=utf-8
import string
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical, normalize


def read_data_file(fname, target_index=0, normalize=True, binary=False):
    content = pd.read_csv(fname, header=None, index_col=False)
    content.dropna(inplace=True)
    content.reset_index(inplace=True, drop=True)

    x = content.ix[:, content.shape[1]-1]
    x = np.array(x)

    # y = content[0] - 1
    # y = to_categorical(y)

    y = content.ix[:, target_index].values
    if normalize:
        max_y = np.max(np.abs(y))
        y = y / max_y
    if binary:
        vals = list(set(y))
        if len(vals) > 2:
            raise Exception("Binary input data is not binary! Dataset %s, target_index=%d" % (fname, target_index) )
        y = np.array([ 0 if a == vals[0] else 1 for a in y ])

    return x, y


def load_restoclub_data():
    train_data = read_data_file('train.csv')
    test_data = read_data_file('test.csv')

    return train_data, test_data


def load_ok_data_gender_normalized():
    train_data = read_data_file('data/ok/ok_train_normalized.csv', target_index=2, binary=True)
    test_data = read_data_file('data/ok/ok_test_normalized.csv', target_index=2, binary=True)

    return train_data, test_data


def load_ok_user_data_gender_normalized():
    train_data = read_data_file('data/ok/ok_user_train_normalized.csv', target_index=2, binary=True)
    test_data = read_data_file('data/ok/ok_user_test_normalized.csv', target_index=2, binary=True)

    return train_data, test_data


def load_ok_data_age_normalized():
    train_data = read_data_file('data/ok/ok_train_normalized.csv', target_index=1, binary=False)
    test_data = read_data_file('data/ok/ok_test_normalized.csv', target_index=1, binary=False)
    return train_data, test_data


def load_ok_user_data_age_normalized():
    train_data = read_data_file('data/ok/ok_user_train_normalized.csv', target_index=1, binary=False)
    test_data = read_data_file('data/ok/ok_user_test_normalized.csv', target_index=1, binary=False)
    return train_data, test_data


def load_ok_data_gender():
    train_data = read_data_file('data/ok/ok_train.csv', target_index=2, binary=True)
    test_data = read_data_file('data/ok/ok_test.csv', target_index=2, binary=True)

    return train_data, test_data


def load_ok_user_data_gender():
    train_data = read_data_file('data/ok/ok_user_train.csv', target_index=2, binary=True)
    test_data = read_data_file('data/ok/ok_user_test.csv', target_index=2, binary=True)

    return train_data, test_data


def load_ok_data_age():
    train_data = read_data_file('data/ok/ok_train.csv', target_index=1, binary=False)
    test_data = read_data_file('data/ok/ok_test.csv', target_index=1, binary=False)
    return train_data, test_data


def load_ok_user_data_age():
    train_data = read_data_file('data/ok/ok_user_train.csv', target_index=1, binary=False)
    test_data = read_data_file('data/ok/ok_user_test.csv', target_index=1, binary=False)
    return train_data, test_data


def load_restoclub_data_with_syns():
    train_data = read_data_file('train_with_syn.csv')
    test_data = read_data_file('test.csv')

    return train_data, test_data


def load_sentirueval_data():
    train_data = read_data_file('data/sentirueval/train.csv')
    test_data = read_data_file('data/sentirueval/test.csv')
    return train_data, test_data


def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):
    for i in range(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
                                 vocab_check)

        yield (input_data, y_sample)


def encode_data(x, maxlen, vocab, vocab_size, check):
    # Iterate over the loaded data and create a matrix of size maxlen x vocabsize
    # In this case that will be 1014x69. This is then placed in a 3D matrix of size
    # data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
    # array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower().replace(' ', ''))
        for c in chars:
            if counter >= 1014:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data


def shuffle_matrix(x, y):
    stacked = np.hstack((np.matrix(x).T, np.asmatrix(y).T))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi


def create_vocab_set():
    # This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = (
    ['й', 'ц', 'у', 'к', 'е', 'н', 'г', 'ш', 'щ', 'з', 'х', 'ъ', 'ф', 'ы', 'в', 'а', 'п', 'р', 'о', 'л', 'д', 'ж', 'э',
     'ё', 'я', 'ч', 'с', 'м', 'и', 'т', 'ь', 'б', 'ю'] +
    list(string.digits) +
    list(string.punctuation) + ['\n'])
    # alphabet = (list(string.ascii_lowercase) + list(string.digits) +
    #             list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check
