# -*- coding:utf-8 -*-

import json
import math
import numpy as np
import string
from random import shuffle

import pandas as pd
from keras.utils.np_utils import to_categorical
from pandas.core.frame import DataFrame

UNKNSYM = u'ξ'
NOSYM = u'ℵ'


def load_embedding_data(env_folder='data'):

    train = pd.read_csv(env_folder + '/embedding/train.csv', header=None, encoding="utf-8")
    train = train.dropna()
    x_train = np.array(train[0])
    y_train = np.array(train[1])

    test = pd.read_csv(env_folder + '/embedding/test.csv', header=None, encoding="utf-8")
    test = test.dropna()
    x_test = np.array(test[0])
    y_test = np.array(test[1])

    return (x_train, y_train), (x_test, y_test)


def prepare_embedding_data(splitting_ratio_train, env_folder):

    with open(env_folder + '/embedding/rawtexts.txt', mode='r') as data_file:
        all_data_list = list(map(lambda x: [x.strip(), x.strip()], data_file.read().split("\n")))

    print("all_data_list", len(all_data_list))

    shuffle(all_data_list)

    splitting = int(math.floor(splitting_ratio_train * len(all_data_list)))
    train_ds = DataFrame(all_data_list[:splitting])
    test_ds = DataFrame(all_data_list[splitting:])

    train_ds.to_csv(env_folder + '/embedding/train.csv', index=False, header=False, sep=",", quotechar='"')
    test_ds.to_csv(env_folder + '/embedding/test.csv', index=False, header=False, sep=",", quotechar='"')


def prepare_restoclub_data(splitting_ratio_train, env_folder):
    """
        Reading, splitting texts and clipped ratings-as-integers
    """

    with open(env_folder + '/restoclub/restoclub.reviews.json', 'r') as data_file:
        json_all_data_list = json.load(data_file, encoding='UTF-8')

    def data_adapter(js_obj):
        """
            NOTA BENE: math.floor for 'total'
        """
        return float(js_obj['ratings']['total']), js_obj['text'].replace("\n", " ").replace("\"\"", "'")

    flat_data = list(map(data_adapter, list(json_all_data_list)))
    splitting = math.floor(splitting_ratio_train * len(flat_data))

    train_ds = DataFrame(flat_data[:splitting])
    test_ds = DataFrame(flat_data[splitting:])

    train_ds.to_csv(env_folder + '/restoclub/train.csv', index=False, header=False, sep=",", quotechar='"')
    test_ds.to_csv(env_folder + '/restoclub/test.csv', index=False, header=False, sep=",", quotechar='"')


def load_restoclub_data(env_folder):
    """
        Loading prepared restoclub texts and clipped ratings-as-integers
    """
    try:
        train = pd.read_csv(env_folder + '/restoclub/train.csv', header=None)
        train = train.dropna()

        x_train = np.array(train[1])
        y_train = train[0]

        test = pd.read_csv(env_folder + '/restoclub/test.csv', header=None)

        x_test = np.array(test[1])
        y_test = test[0]

        return (x_train, y_train), (x_test, y_test)
    except IOError as e:
        print (e)
        prepare_restoclub_data(0.95, env_folder)
        load_restoclub_data(env_folder)


def load_restoclub_data_for_encdec(env_folder):
    """
        Encoding-decoding data
    """
    try:
        train = pd.read_csv(env_folder + '/restoclub/train.csv', header=None)
        train = train.dropna()

        x_train = np.array(train[1])
        y_train = np.array(train[1])

        test = pd.read_csv(env_folder + '/restoclub/test.csv', header=None)
        x_test = np.array(test[1])
        y_test = np.array(test[1])

        return (x_train, y_train), (x_test, y_test)
    except IOError as e:
        print (e)
        prepare_restoclub_data(0.95, env_folder)
        load_restoclub_data(env_folder)


def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen, batch_size):

    for i in range(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        x_for_input = encode_data(x_sample, maxlen, vocab, vocab_size, vocab_check)
        y_for_fitting = encode_data(y_sample, maxlen, vocab, vocab_size, vocab_check)

        yield (x_for_input, y_for_fitting, x_sample, y_sample)


def encode_data(x, maxlen, vocab, vocab_size, check):
    """
        Iterate over the loaded data and create a matrix of size maxlen x vocabsize
        In this case that will be 1014x69. This is then placed in a 3D matrix of size
        data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
        array. Chars not in the vocab are encoded into an all zero vector.
    """
    input_data = np.zeros((len(x), maxlen, vocab_size))

    for dix, sent in enumerate(x):

        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))

        try:
            chars = list(sent.lower())  # .replace(' ', ''))
        except:
            print("ERROR " + str(dix) + " " + str(sent))
            continue

        for c in chars:
            if counter >= maxlen:
                break
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                else:
                    # char not in set, we replace it with special symbol
                    ix = vocab[UNKNSYM]
                    char_array[ix] = 1

                sent_array[counter, :] = char_array
                counter += 1

        input_data[dix, :, :] = sent_array

    return input_data


def decode_data(matrix, reverse_vocab):
    """
        data_samples x maxlen x vocab_size
        Argmaxing each row and applying reversed vocabulary for sequence decoding
    """
    try:
        return "".join([reverse_vocab[np.argmax(row)] for encoded_matrix in matrix for row in encoded_matrix]).strip(
            NOSYM)
    except:
        return "ERROR"


def shuffle_matrix(x, y):
    """
        Joint random sort
    """
    stacked = np.hstack((np.matrix(x).T, np.matrix(y).T))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1]).flatten()

    return xi, yi


def create_vocab_set():
    alphabet = \
        (list(NOSYM + u"qwertyuiopasdfghjklzxcvbnmёйцукенгшщзхъфывапролджэячсмитьбю«»…–“”№—") +
         list(string.digits) +
         list(string.punctuation) +
         ['\n', ' ', UNKNSYM])

    vocab_size = len(alphabet) + 1
    check = set(alphabet)
    vocab = {}
    reverse_vocab = {}

    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check


if __name__ == '__main__':
    prepare_restoclub_data(0.7, "data")
# parser.add_argument('--spl', type=float,
#                     default=0.9,
#                     help='dataset splitting train/test, float')
