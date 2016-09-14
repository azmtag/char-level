# coding:utf-8

import numpy as np
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.optimizers import Adam

np.random.seed(2)


def build_model(sequence_length, embedding_dim, filter_sizes, num_filters,
                dropout_prob, hidden_dims, model_variation, vocabulary,
                embedding_weights,
                subsample_length=1, pool_length=2, mode='1mse'):
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []

    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=subsample_length)(graph_in)
        pool = MaxPooling1D(pool_length=pool_length)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()

    if not model_variation == 'CNN-static':
        model.add(Embedding(len(vocabulary),
                            embedding_dim,
                            input_length=sequence_length,
                            weights=embedding_weights))

    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))

    if mode == '1mse':
        adam = Adam()
        model.compile(loss='mean_square_error', optimizer=adam, metrics=['accuracy'])
    else:
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model
