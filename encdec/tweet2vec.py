# -*- coding:utf-8 -*-

from keras.layers import Input, LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import RepeatVector
from keras.models import Model
from keras.optimizers import Adam


def model(filter_kernels, timesteps, vocab_size, nb_filter, latent_dim_lstm_enc, latent_dim_lstm_dec):
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

    encoded = LSTM(output_dim=latent_dim_lstm_enc, return_sequences=False)(conv3)

    encoded_copied = RepeatVector(n=timesteps)(encoded)

    predecoded = LSTM(output_dim=latent_dim_lstm_dec, return_sequences=True)(encoded_copied)

    decoded = LSTM(output_dim=vocab_size,
                   return_sequences=True,
                   activation='softmax')(predecoded)

    sequence_autoencoder = Model(inputs, decoded)

    adam = Adam()
    sequence_autoencoder.compile(loss='categorical_crossentropy', optimizer=adam)

    """
    memo: How to get encoder only? Simply do after training:
        encoder = Model(input=inputs, output=encoded)
        X_encoded = encoder.predict(X)
    """
    encoder_only_model = Model(input=inputs, output=encoded)

    return sequence_autoencoder, encoder_only_model
