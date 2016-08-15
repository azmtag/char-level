# -*- coding:utf-8 -*-

"""
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
"""

from __future__ import print_function
from __future__ import division
import os
import json
import tweet2vec
import datetime
import numpy as np
import data_helpers
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import logging

logging.basicConfig(filename='all_results.log',
                    format='[%(asctime)s] %(name)s | %(levelname)s - %(message)s',
                    level=logging.DEBUG)

lg = logging.getLogger("L")
lg.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
lg.addHandler(ch)

# for reproducibility
np.random.seed(123)


def get_session(gpu_fraction=0.2):
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


KTF.set_session(get_session())

# set parameters:

subset = None

# Whether to save model parameters

save = True
autoencoder_model_name_path = 'params/t2v_model.json'
autoencoder_model_weights_path = 'params/t2v_model_weights.h5'
encoder_model_name_path = 'params/et2v_model.json'
encoder_model_weights_path = 'params/et2v_model_weights.h5'

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 140

# Filters for conv layers
nb_filter = 512

# Number of units in the dense layer
# dense_outputs = 1024

# Conv layer kernel size
filter_kernels = [7, 7, 3, 3]

# Compile/fit params
batch_size = 200
test_batch_size = 100
nb_epoch = 10

# LSTM latent vector size, enc_N from the paper
# latent_dim = 256
latent_dim = 450


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
                                                   latent_dim)

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
        # todo: print embedding https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
        lg.info('----------------------------------------------------------------')

    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    lg.info('Epoch {}. Loss: {}.\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, e_elap, t_elap))

if save:
    lg.info('Saving model params...')
    json_string = autoencoder_model.to_json()

    with open(autoencoder_model_name_path, 'w') as f:
        json.dump(json_string, f, ensure_ascii=False)

    autoencoder_model.save_weights(autoencoder_model_name_path, overwrite=True)
