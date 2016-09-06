# coding=utf-8

import string
import pandas
import numpy
import json
import datetime

from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from enum import Enum

#------------------------------------#

def english_alphabet():
    return list(string.ascii_lowercase)

#------------------------------------#

class CrepeModel(object):

    #------------------------------------#

    class DataFileType(Enum):
        json = 0
        csv = 1

    #------------------------------------#

    # TODO: make difference between regression and classification

    def __init__(self, alphabet=english_alphabet(), filter_numbers=256,
                 dense_outputs=1024, filter_kernels=[7, 7, 3, 3, 3, 3],
                 maxlen=None, categories=None):
        """

        :param alphabet:
            List of letters that will be used to encoded. To that list will be
            appended digits, punctuation and new line symbol. Other symbols
            will be skipped.
        :param filter_numbers:

        :param dense_outputs:
        :param filter_kernels:
        :return:
        """

        # build dictionary from alphabet: letter => position
        assert type(alphabet) == list
        self.__build_alphabet(alphabet)

        self.__filter_numbers = filter_numbers
        self.__dense_outputs = dense_outputs
        self.__filter_kernels = filter_kernels
        self.__maxlen = maxlen
        self.__categories = categories
        self.__batch_size = 128
        self.__finish = False  # mark that model is not ready to save

        self.__x_train = None
        self.__y_train = None
        self.__x_test = None
        self.__y_test = None

    #------------------------------------#

    def __build_alphabet(self, alphabet):
        letters = alphabet + list(string.digits) + list(string.punctuation) + ['\n']

        self.__alphabet = {letter: inx for inx, letter in enumerate(letters)}

    #------------------------------------#

    def load_data(self, path_to_data, path_to_test_data=None, formatter=None, file_type=DataFileType.json):
        """

        :param path_to_data:
            Path to file with data. There is could be only data for train if path_to_test_data
            is not specified.
        :param path_to_test_data:
            Path to file with data for test. If it is not specified then data from file for train
            will be shuffled and split to 80/20.
        :param formatter:
            Functor that takes pandas DataFrame as parameter and format it to (x, y) for train/test.
        :param file_type:
            File to type.
        :return:
        """
        print("Loading and preparing data start...")
        data, test_data = None, None
        x_data, y_data = None, None
        x_test, y_test = None, None

        # case of json type file
        if file_type == CrepeModel.DataFileType.json:
            print("Loading data...")
            data = pandas.read_json(path_to_data)

            if path_to_test_data:
                print("Loading test data...")
                test_data = pandas.read_json(path_to_test_data)

        # case of csv type file
        elif file_type == CrepeModel.DataFileType.csv:
            print("Loading data...")
            data = pandas.read_csv(path_to_data, header=None)

            if path_to_test_data:
                print("Loading test data...")
                test_data = pandas.read_csv(path_to_test_data, header=None)
        else:
            raise Exception("Set wrong file type to read.")

        data = data.dropna()
        if test_data is not None:
            test_data = test_data.dropna()

        # format the data from path_to_data file
        # if formatter is None, then expected that first column is data, the second - class
        print("Data formatting...")
        if not formatter:
            x_data = numpy.array(data[0])
            y_data = to_categorical(data[1])

            if test_data:
                x_test = numpy.array(test_data[0])
                y_test = to_categorical(test_data[1])
        else:
            x_data, y_data = formatter(data)

            if test_data is not None:
                x_test, y_test = formatter(test_data)

        # if path to test data is not specified then get all data, shuffle it, and split to 80/20
        if test_data is None:
            # shuffle
            print("Test data not found, then shuffling and splitting...")
            x_data, y_data = self.__shuffle(x_data, y_data)

            # split data
            split_index = int(len(x_data) * 0.8)

            self.__x_train = x_data[:split_index]
            self.__y_train = y_data[:split_index]
            self.__x_test = x_data[split_index:]
            self.__y_test = y_data[split_index:]
        else:
            # already split
            self.__x_train = x_data
            self.__y_train = y_data
            self.__x_test = x_test
            self.__y_test = y_test

        # try to deduce maxlen if it is not specified
        if not self.__maxlen:
            len_func = numpy.vectorize(len)
            lengths = len_func(self.__x_train)
            self.__maxlen = int(lengths.mean() + 3 * lengths.std())

            print("Set 'maxlen' to %s" % self.__maxlen)

        print("Preparing data finished. Lines to train: %s. Lines to test: %s" % (len(self.__x_train), len(self.__x_test)))

    #------------------------------------#

    def run(self, epochs=10, batch_size=128):
        if not self.__maxlen:
            raise Exception("Max len isn't specified for the model")

        if self.__batch_size != batch_size:
            self.__batch_size = batch_size

        if self.__batch_size == 0:
            raise Exception("Batch size couldn't be 0!")

        print("Run starting...")

        # building a net
        self.__finish = False # mark that model is not ready save
        self.__build()

        start_time = datetime.datetime.now()
        log_step = int((len(self.__x_train) / self.__batch_size) * 0.1)

        for epoch in range(epochs):
            epoch_time = datetime.datetime.now()
            x_train, y_train = self.__shuffle(self.__x_train, self.__y_train)
            x_test, y_test = self.__shuffle(self.__x_test, self.__y_test)

            train_batch = self.__batch_generator(x_train, y_train)
            test_batch = self.__batch_generator(x_test, y_test)

            # TRAIN PART STARTS #
            train_accuracy, train_loss = 0.0, 0.0
            train_step, batch_step = 1, 1

            batch_time = datetime.datetime.now()
            for x_batch, y_batch in train_batch:
                train_result = self.__model.train_on_batch(x_batch, y_batch)

                # add result from this batch
                train_loss += train_result[0]
                train_accuracy += train_result[1]

                # calculate average
                train_loss_avg = train_loss / train_step
                train_accuracy_avg = train_accuracy / train_step

                # each 10%
                if train_step % log_step == 0:
                    print("Batch percent: %s" % (str(10 * batch_step) + '%'))
                    print("Average: %s - loss, %s - accuracy" % (train_loss_avg, train_accuracy_avg))
                    print("Batch time: %s" % (datetime.datetime.now() - batch_time))
                    batch_time = datetime.datetime.now()
                    batch_step += 1

                train_step += 1
            # TRAIN PART ENDS #

            # TEST PART STARTS #
            test_accuracy, test_loss = 0.0, 0.0
            test_accuracy_avg, test_loss_avg = 0.0, 0.0
            test_step = 1

            for x_batch, y_batch in test_batch:
                test_result = self.__model.test_on_batch(x_batch, y_batch)

                # add result from this batch
                test_loss += test_result[0]
                test_accuracy += test_result[1]

                # calculate average
                test_loss_avg = test_loss / test_step
                test_accuracy_avg = test_accuracy / test_step

                test_step += 1

            # TEST PART ENDS #

            print('Epoch %s, Average: %s - loss, %s - accuracy' % (epoch, test_loss_avg, test_accuracy_avg))
            print("Epoch time: %s" % (datetime.datetime.now() - epoch_time))

        print("Run time: %s" % (datetime.datetime.now() - start_time))

        self.__finish = True # mark that model is ready to save
        print("Run finished.")

    #------------------------------------#

    def save(self, file_to_save):
        if not self.__finish:
            raise Exception("Model is not ready to save or empty yet.")

        print("Saving model params...")

        json_string = self.__model.to_json()
        with open(file_to_save, 'w') as fout:
            json.dump(json_string, fout)

        self.__model.save_weights(file_to_save + '.h5')

        print("Saving finished.")

    #------------------------------------#

    def __batch_generator(self, x, y):
        for inx in range(0, len(x), self.__batch_size):
            x_sample = x[inx:inx + self.__batch_size]
            y_sample = y[inx:inx + self.__batch_size]

            data = self.__encdoing(x_sample)

            yield (data, y_sample)

    #------------------------------------#

    def __shuffle(self, x, y):
        stacked = numpy.hstack((numpy.matrix(x).T, y))
        numpy.random.shuffle(stacked)
        return numpy.array(stacked[:, 0]).flatten(), numpy.array(stacked[:, 1:])

    #------------------------------------#

    def __encdoing(self, raw_data):
        alphabet_size = len(self.__alphabet)

        data = numpy.zeros((len(raw_data), self.__maxlen, alphabet_size))

        # iterate through the sentence in data
        for sentence_pos, sentence in enumerate(raw_data):
            char_pos = 0
            sentence_array = numpy.zeros((self.__maxlen, alphabet_size))
            chars = list(sentence.lower().replace(' ', ''))[:self.__maxlen]

            # run through the chars in sentence and make one-hot coding each char
            for char in chars:
                char_array = numpy.zeros(alphabet_size, dtype=numpy.int)

                if char in self.__alphabet:
                    char_array[self.__alphabet[char]] = 1

                sentence_array[char_pos, :] = char_array
                char_pos += 1

            data[sentence_pos, :, :] = sentence_array

        return data

    #------------------------------------#

    def __build(self):
        print("Start building the model...")
        alphabet_size = len(self.__alphabet)

        #Define what the input shape looks like
        inputs = Input(shape=(self.__maxlen, alphabet_size), name='input', dtype='float32')

        #All the convolutional layers...
        conv = Convolution1D(nb_filter=self.__filter_numbers,
                             filter_length=self.__filter_kernels[0],
                             border_mode='valid', activation='relu',
                             input_shape=(self.__maxlen, alphabet_size))(inputs)

        conv = MaxPooling1D(pool_length=3)(conv)

        conv1 = Convolution1D(nb_filter=self.__filter_numbers,
                              filter_length=self.__filter_kernels[1],
                              border_mode='valid',
                              activation='relu')(conv)

        conv1 = MaxPooling1D(pool_length=3)(conv1)

        conv2 = Convolution1D(nb_filter=self.__filter_numbers,
                              filter_length=self.__filter_kernels[2],
                              border_mode='valid',
                              activation='relu')(conv1)

        conv3 = Convolution1D(nb_filter=self.__filter_numbers,
                              filter_length=self.__filter_kernels[3],
                              border_mode='valid',
                              activation='relu')(conv2)

        conv4 = Convolution1D(nb_filter=self.__filter_numbers,
                              filter_length=self.__filter_kernels[4],
                              border_mode='valid',
                              activation='relu')(conv3)

        conv5 = Convolution1D(nb_filter=self.__filter_numbers,
                              filter_length=self.__filter_kernels[5],
                              border_mode='valid',
                              activation='relu')(conv4)

        conv5 = MaxPooling1D(pool_length=3)(conv5)
        conv5 = Flatten()(conv5)

        #Two dense layers with dropout of .5
        z = Dropout(0.5)(Dense(self.__dense_outputs, activation='relu')(conv5))
        z = Dropout(0.5)(Dense(self.__dense_outputs, activation='relu')(z))

        #Output dense layer with softmax activation
        pred = Dense(self.__categories, activation='softmax', name='output')(z)

        self.__model = Model(input=inputs, output=pred)

        sgd = SGD(lr=0.01, momentum=0.9)
        self.__model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['accuracy'])

        print("Building model finished")

    #------------------------------------#

#------------------------------------#

if __name__ == '__main__':
    DEBUG = False

    #--------------------------#
    if DEBUG:
        def formatter(data):
            x = data[1] + data[2]
            y = data[0] - 1

            return (numpy.array(x), to_categorical(y))

        model = CrepeModel(categories=4)
        model.load_data(path_to_data='data/train.csv',
                        path_to_test_data='data/test.csv',
                        formatter=formatter,
                        file_type=CrepeModel.DataFileType.csv)

        model.run()
    #--------------------------#

    def formtatter(data):
        total = data[1]
        mean = data[1].mean()
        total_data = data[1].apply(lambda x: 0 if x < mean else 1)

        x = data[0]
        y = to_categorical(total_data)

        return (numpy.array(x), numpy.array(y))

    # def rus alphabet
    def rus_alphabet():
        return list(u"абвгдеёжзийклмнопрстуфчцшщъыьэюя")

    model = CrepeModel(alphabet=rus_alphabet(), categories=2)
    model.load_data(path_to_data="data/train.csv",
                    path_to_test_data="data/test.csv",
                    formatter=formtatter,
                    file_type=CrepeModel.DataFileType.csv)

    model.run()