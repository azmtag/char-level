# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from keras.layers.wrappers import TimeDistributed
import datetime
import json
import logging
import numpy as np
import os

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.layers import Input, Dense
from keras.layers.core import RepeatVector
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Model
from keras.optimizers import Adam

import data_helpers

logging.basicConfig(filename='all_results.log',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

lg = logging.getLogger("ConsoleLogger")
lg.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
lg.addHandler(ch)


def model(maxlen, vocab_size, latent_dim):
    """
        Simple autoencoder for sequences
    """

    input_dim = vocab_size
    timesteps = maxlen

    lg.info("input_dim " + str(vocab_size) +
            " latent_dim " + str(latent_dim) +
            " timesteps " + str(timesteps))

    inputs = Input(shape=(timesteps, input_dim))

    lg.info("Input set. " + str(inputs))
    lg.info("Setting encoder: out-dim " + str(latent_dim))

    # takes time :[
    #encoded_0 = LSTM(latent_dim,
    #               # encoded = SimpleRNN(latent_dim,
    #               activation='relu',
    #               return_sequences=True)(inputs)
    encoded = LSTM(latent_dim,
                   # encoded = SimpleRNN(latent_dim,
                   activation='relu',
                   return_sequences=False)(inputs) #encoded)
    # encoded = LSTM(latent_dim, return_sequences=False)(inputs)

    lg.info("Encoder set: " + str(encoded))

    #encoded2 = Dense(latent_dim / 3 * 2, activation='sigmoid')

    repeated_embedding = RepeatVector(timesteps)(encoded)

    lg.info("Repeated embedding added: " + str(repeated_embedding))
    lg.info("Setting decoder")

    # takes time :[
    decoded = LSTM(input_dim,
                   # decoded = SimpleRNN(input_dim,
                   #inner_init='identity',
                   return_sequences=True,
                   activation='relu')(repeated_embedding)

    decoded_res = TimeDistributed(Dense(input_dim, activation='softmax'))(decoded)

    lg.info("Decoder added: " + str(decoded_res))

    # reshaped_decoder = Reshape(target_shape=(1, input_dim * timesteps))(decoded)
    # sequence_autoencoder = Model(inputs, reshaped_decoder)
    sequence_autoencoder = Model(inputs, decoded_res)

    lg.info("Autoencoder brought together as a model: " + str(sequence_autoencoder))

    adam = Adam()

    lg.info("Adam optimizer created: " + str(adam))
    lg.info("Model constructed, compiling now...")

    sequence_autoencoder.compile(loss='categorical_crossentropy', optimizer=adam)

    lg.info("Model compiled.")

    return sequence_autoencoder


def get_session(gpu_fraction=0.2):
    """
        Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# for reproducibility
np.random.seed(123)
KTF.set_session(get_session())

# set parameters:
subset = None

# Whether to save model parameters
save = True
model_name_path = 'params/lstm_dumb_model.json'
model_weights_path = 'params/lstm_dumb_model_weights.h5'

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 7

# Compile/fit params
batch_size = 1000
test_batch_size = 50
nb_epoch = 50

representation_dim = 450

lg.info('Loading data...')

# Expect x to be a list of sentences. y to also be a list of sentences
(x_train, y_train), (x_test, y_test) = data_helpers.load_embedding_data()

lg.info('Creating vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

lg.info('Vocabulary: ' + ",".join(vocab))
# test_data = data_helpers.encode_data(x_test, maxlen, vocab, vocab_size, check)

lg.info('Build model...')
dumb_model = model(maxlen, vocab_size, representation_dim)

lg.info('Fit model...')
initial = datetime.datetime.now()

for e in range(nb_epoch):

    xi, yi = data_helpers.shuffle_matrix(x_train, y_train)
    xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)

    if subset:
        batches = data_helpers.mini_batch_generator(xi[:subset], yi[:subset],
                                                    vocab, vocab_size, check,
                                                    maxlen, batch_size=batch_size)
    else:
        batches = data_helpers.mini_batch_generator(xi, yi, vocab, vocab_size,
                                                    check, maxlen, batch_size=batch_size)

    test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                     vocab_size, check, maxlen,
                                                     batch_size=test_batch_size)

    loss = 0.0
    step = 1
    start = datetime.datetime.now()

    lg.info('EPOCH: {}'.format(e))
    lg.info('Training started')

    for x_train_batch, y_train_batch, _, _ in batches:

        # lg.info('Training on batch ' + str(x_train_batch.shape) + ' -> ' + str(y_train_batch.shape))

        f = dumb_model.train_on_batch(x_train_batch, y_train_batch)
        loss += f
        loss_avg = loss / step

        if step % 50 == 0:
            lg.info('Train step: {}, loss: {}'.format(step, loss_avg))
        step += 1

    test_loss = 0.0
    test_loss_avg = 0.0
    test_step = 1

    lg.info('Testing started ----------------------------------')

    for x_test_batch, y_test_batch, x_text, y_text in test_batches:
        # lg.info('Testing on batch ' + str(x_test_batch.shape) + ' -> ' + str(y_test_batch.shape))

        f_ev = dumb_model.test_on_batch(x_test_batch, y_test_batch)

        test_loss += f_ev
        test_loss_avg = test_loss / test_step
        test_step += 1

        if test_step % 10 == 0:
            lg.info('Test step: {}, loss: {}'.format(test_step, test_loss_avg))
            predicted_seq = dumb_model.predict(np.array([x_test_batch[0]]))
            lg.info(
            'Shapes x {} y_true {} y_pred {}'.format(x_test_batch[0].shape, y_test_batch[0].shape, predicted_seq.shape))
            lg.info('Input:    \t[' + x_text[0][:maxlen] + "]")
            lg.info(u'Predicted:\t[' + data_helpers.decode_data(predicted_seq, reverse_vocab) + "]")
            # todo: print embedding https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
            lg.info('----------------------------------------------------------------')

    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial

    lg.info('Epoch {}. Loss: {}.\nEpoch time: {}. Total time: {}\n\n\n'.format(e, test_loss_avg, e_elap, t_elap))

if save:
    lg.info('Saving model params...')
    json_string = dumb_model.to_json()

    with open(model_name_path, 'w') as f:
        json.dump(json_string, f, ensure_ascii=False)

    dumb_model.save_weights(model_weights_path, overwrite=True)
