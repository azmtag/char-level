# -*- coding:utf-8 -*-

from keras.layers import Input, LSTM, Dense
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import RepeatVector
from keras.layers.recurrent import GRU, SimpleRNN
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed


def model(filter_kernels,
          timesteps,
          vocab_size,
          nb_filter,
          latent_dim_lstm_enc,
          latent_dim_lstm_dec,
          recurrent_model):
    # Define what the input shape looks like
    inputs = Input(shape=(timesteps, vocab_size), name='input', dtype='float32')

    # All the convolutional layers...
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(timesteps, vocab_size))(inputs)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu')(conv2)

    # Two dense layers with dropout of .5
    # z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    # z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    # todo: convert to map
    if recurrent_model == 'GRU':
        rec_layer_f = GRU
    elif recurrent_model == 'LSTM':
        rec_layer_f = LSTM
    elif recurrent_model == 'SimpleRNN':
        rec_layer_f = SimpleRNN
    else:
        raise Exception('No such model ' + str(recurrent_model))

    encoded = rec_layer_f(output_dim=latent_dim_lstm_enc, return_sequences=False)(conv3)

    # if recurrent_model == 'GRU':
    #     encoded = GRU(output_dim=latent_dim_lstm_enc, return_sequences=False)(conv3)
    # elif recurrent_model == 'LSTM':
    #     encoded = LSTM(output_dim=latent_dim_lstm_enc, return_sequences=False)(conv3)
    # elif recurrent_model == 'SimpleRNN':
    #     encoded = SimpleRNN(output_dim=latent_dim_lstm_enc, return_sequences=False)(conv3)
    # else:
    #     raise Exception('No such model ' + str(recurrent_model))

    encoded_copied = RepeatVector(n=timesteps)(encoded)

    predecoded = rec_layer_f(output_dim=latent_dim_lstm_dec, return_sequences=True)(encoded_copied)

    # if recurrent_model == 'GRU':
    #     predecoded = GRU(output_dim=latent_dim_lstm_dec, return_sequences=True)(encoded_copied)
    # elif recurrent_model == 'LSTM':
    #     predecoded = LSTM(output_dim=latent_dim_lstm_dec, return_sequences=True)(encoded_copied)

    decoded = rec_layer_f(output_dim=vocab_size,
                          return_sequences=True,
                          activation='relu')(predecoded)

    # if recurrent_model == 'GRU':
    #     decoded = GRU(output_dim=vocab_size,
    #                   return_sequences=True,
    #                   activation='relu')(predecoded)
    # else:
    #     decoded = LSTM(output_dim=vocab_size,
    #                    return_sequences=True,
    #                    activation='relu')(predecoded)

    decoded_res = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoded)

    sequence_autoencoder = Model(inputs, decoded_res)

    adam = Adam()
    sequence_autoencoder.compile(loss='categorical_crossentropy', optimizer=adam)

    encoder_only_model = Model(input=inputs, output=encoded)

    return sequence_autoencoder, encoder_only_model
