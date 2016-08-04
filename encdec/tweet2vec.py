# -*- coding:utf-8 -*-

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import RepeatVector
from keras.models import Model
from keras.optimizers import Adam


def model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, latent_dim_lstm):
    # Define what the input shape looks like
    inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    # All the convolutional layers...
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(inputs)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu')(conv2)

    # conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
    #                       border_mode='valid', activation='relu')(conv3)
    #
    # conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
    #                       border_mode='valid', activation='relu')(conv4)
    # conv5 = BatchNormalization()(conv5)

    # conv5 = MaxPooling1D(pool_length=3)(conv5)
    # conv5 = Flatten()(conv5)

    # Two dense layers with dropout of .5
    # z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    # z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    encoded = LSTM(latent_dim_lstm)(conv3)
    decoded = RepeatVector(maxlen)(encoded)
    decoded = LSTM(maxlen, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)

    adam = Adam()
    sequence_autoencoder.compile(loss='categorical_crossentropy', optimizer=adam)

    return sequence_autoencoder
