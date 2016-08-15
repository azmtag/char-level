# -*- coding:utf-8 -*-

"""
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
"""

from __future__ import division
from __future__ import print_function

import argparse as ap
import datetime
import json
import logging
import os

import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf

import data_helpers
import tweet2vec

parser = ap.ArgumentParser(description='Params for tweet2vec [BTW, they define saved model file name]')

parser.add_argument('--epochs', type=int,
                    default=300,
                    help='default=300; epochs count')

parser.add_argument('--maxlen', type=int,
                    default=140,
                    help='default=140; max sequence length')

parser.add_argument('--rnn', type=str, choices=['SimpleRNN', 'LSTM', 'GRU'],
                    default='SimpleRNN',
                    help='default=SimpleRNN; recurrent layers type')

parser.add_argument('--rnndim', type=int,
                    default=256,
                    help='default=256; recurrent layers dimensionality')

parser.add_argument('--batch', type=int,
                    default=80,
                    help='default=80; training batch size')

parser.add_argument('--test_batch', type=int,
                    default=40,
                    help='default=40; validation batch size')

parser.add_argument('--gpu_fraction', type=float,
                    default=0.2,
                    help='default=0.2; GPU fraction, please, use with care')

args = parser.parse_args()

# setting model params
maxlen = args.maxlen
latent_dim = args.rnndim
rnn_type = args.rnn

# Compile/fit params
batch_size = args.batch
test_batch_size = args.test_batch
nb_epoch = args.epochs
gpu_fraction_default = args.gpu_fraction

# for reproducibility
np.random.seed(123)

config_name = '_maxlen_' + str(maxlen) + \
              '_rnn_' + rnn_type + \
              '_rnndim_' + str(latent_dim) + \
              '_batch_' + str(batch_size) + \
              '_epochs_' + str(nb_epoch)

autoencoder_model_name_path = 'params/encdec' + config_name + '.json'
autoencoder_model_weights_path = 'params/encdec' + config_name + '_weights.h5'
encoder_model_name_path = 'params/t2v' + config_name + '.json'
encoder_model_weights_path = 'params/t2v' + config_name + '_weights.h5'

logging.basicConfig(filename='all' + config_name + '.log',
                    format='[%(asctime)s] %(name)s | %(levelname)s - %(message)s',
                    level=logging.DEBUG)

lg = logging.getLogger("LGR")
lg.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
lg.addHandler(ch)


def get_session(gpu_fraction):
    """
        Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session(gpu_fraction=gpu_fraction_default))

# set parameters:
subset = None

# Whether to save model parameters
save = True

# Filters for conv layers
nb_filter = 512

# Conv layer kernel size
filter_kernels = [7, 7, 3, 3]

lg.info('Loading data...')

# Expect x to be a list of sentences. Y to be a one-hot encoding of the categories.
(xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data_for_encdec("data")

lg.info('Creating vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

lg.info(str(vocab))
test_data = data_helpers.encode_data(x_test, maxlen, vocab, vocab_size, check)

lg.info('Build model...')
autoencoder_model, encoder_model = tweet2vec.model(filter_kernels,
                                                   maxlen,
                                                   vocab_size,
                                                   nb_filter,
                                                   latent_dim,
                                                   latent_dim,
                                                   rnn_type)

lg.info('Fit model...')
initial = datetime.datetime.now()

for e in range(nb_epoch):

    xi, yi = data_helpers.shuffle_matrix(xt, yt)
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

    # accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    lg.info('Epoch: {}'.format(e))

    for x_train, y_train, x_t, y_t in batches:

        # todo: synonym-replaced texts dataset needed
        f = autoencoder_model.train_on_batch(x_train, y_train)
        loss += f
        loss_avg = loss / step

        if step % 100 == 0:
            lg.info('Step: {}'.format(step))
            # lg.info('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
            lg.info('Loss: {}.'.format(loss_avg))
        step += 1

    # test_accuracy = 0.0
    test_loss = 0.0
    test_step = 1
    test_loss_avg = 0.0

    for x_test_batch, y_test_batch, x_text, y_text in test_batches:
        # todo: synonym-replaced texts dataset needed
        f_ev = autoencoder_model.test_on_batch(x_test_batch, y_test_batch)
        test_loss += f_ev  # [0]
        test_loss_avg = test_loss / test_step
        test_step += 1

        lg.info('Test step: {}, loss: {}'.format(test_step, test_loss_avg))
        predicted_seq = autoencoder_model.predict(np.array([x_test_batch[0]]))
        lg.info(
            'Shapes x {} y_true {} y_pred {}'.format(
                x_test_batch[0].shape,
                y_test_batch[0].shape,
                predicted_seq.shape))
        lg.info('Input:    \t[' + x_text[0][:maxlen] + "]")
        lg.info(u'Predicted:\t[' + data_helpers.decode_data(predicted_seq, reverse_vocab) + "]")
        # todo: print embedding
        # todo: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
        lg.info('----------------------------------------------------------------')

    stop = datetime.datetime.now()

    e_elap = stop - start
    t_elap = stop - initial
    lg.info('Epoch {}. Loss: {}.\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, e_elap, t_elap))

if save:
    lg.info('Saving encdec params...')
    json_string = autoencoder_model.to_json()

    with open(autoencoder_model_name_path, 'w') as f:
        json.dump(json_string, f, ensure_ascii=False)

    autoencoder_model.save_weights(autoencoder_model_weights_path, overwrite=True)

    lg.info('Saving encoder')

    json_string = encoder_model.to_json()

    with open(encoder_model_name_path, 'w') as f:
        json.dump(json_string, f, ensure_ascii=False)

    encoder_model.save_weights(encoder_model_weights_path, overwrite=True)
