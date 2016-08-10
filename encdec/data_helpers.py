# -*- coding:utf-8 -*-

import json
import math
import numpy as np
import string

import pandas as pd
from keras.utils.np_utils import to_categorical
from pandas.core.frame import DataFrame
from random import shuffle

UNKNSYM = u'ξ'


def load_ag_data():
    train = pd.read_csv('data/ag_news_csv/train.csv', header=None)
    train = train.dropna()

    x_train = train[1] + train[2]
    x_train = np.array(x_train)

    y_train = train[0] - 1
    y_train = to_categorical(y_train)

    test = pd.read_csv('data/ag_news_csv/test.csv', header=None)
    x_test = test[1] + test[2]
    x_test = np.array(x_test)

    y_test = test[0] - 1
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


def load_embedding_data():

    train = pd.read_csv('data/embedding/train.csv', header=None)
    train = train.dropna()
    x_train = np.array(train[0])
    y_train = np.array(train[1])

    test = pd.read_csv('data/embedding/test.csv', header=None)
    test = test.dropna()
    x_test = np.array(test[0])
    y_test = np.array(test[1])

    return (x_train, y_train), (x_test, y_test)


def prepare_embedding_data(splitting_ratio_train, env_folder):
    all_data_list = []

    with open(env_folder + '/embedding/rawtexts.txt', mode='r') as data_file:
        all_data_list = map(lambda x: [x.strip(), x.strip()], data_file.read().split("\n"))

    shuffle(all_data_list)

    splitting = int(math.floor(splitting_ratio_train * len(all_data_list)))
    train_ds = DataFrame(all_data_list[:splitting])
    test_ds = DataFrame(all_data_list[splitting:])

    train_ds.to_csv(env_folder + '/embedding/train.csv', index=False, header=False, sep=",", quotechar='"')
    test_ds.to_csv(env_folder + '/embedding/test.csv', index=False, header=False, sep=",", quotechar='"')


def prepare_restoclub_data(splitting_ratio_train, env_folder):
    json_all_data_list = []

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
    try:
        train = pd.read_csv(env_folder + '/restoclub/train.csv', header=None)
        train = train.dropna()

        x_train = train[1]
        x_train = np.array(x_train)

        y_train = train[0]

        print(x_train.shape)
        print(y_train.shape)

        test = pd.read_csv(env_folder + '/restoclub/test.csv', header=None)
        x_test = test[1]
        x_test = np.array(x_test)

        y_test = test[0]

        return (x_train, y_train), (x_test, y_test)
    except IOError as e:
        print (e)
        prepare_restoclub_data(0.95, env_folder)
        load_restoclub_data(env_folder)


def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen, batch_size=128):

    for i in range(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data = encode_data(x_sample, maxlen, vocab, vocab_size, vocab_check)
        y_for_fitting = encode_data(y_sample, maxlen, vocab, vocab_size, vocab_check)

        yield (input_data, y_for_fitting)


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


def shuffle_matrix(x, y):

    stacked = np.hstack((np.matrix(x).T, np.matrix(y).T))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1]).flatten()

    return xi, yi


def create_vocab_set():
    alphabet = \
        (list(u"qwertyuiopasdfghjklzxcvbnmёйцукенгшщзхъфывапролджэячсмитьбю«»…–“”№—") +
         list(string.digits) +
         list(string.punctuation) +
         ['\n', ' ', UNKNSYM])

    print("ALPHABET: " + ",".join(alphabet))

    vocab_size = len(alphabet) + 1
    check = set(alphabet)

    print("CHECK:" + ",".join(check))

    vocab = {}
    reverse_vocab = {}

    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check


if __name__ == '__main__':
    # loading test
    prepare_embedding_data(0.95, "data")
