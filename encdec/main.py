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
import logging as lg

lg.basicConfig(filename='all_results.log',
               format='[%(asctime)s] %(name)s | %(levelname)s - %(message)s',
               level=lg.DEBUG)

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
model_name_path = 'params/crepe_model.json'
model_weights_path = 'params/crepe_model_weights.h5'

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 1014

# Filters for conv layers
nb_filter = 512

# Number of units in the dense layer
dense_outputs = 1024

# Conv layer kernel size
filter_kernels = [7, 7, 3, 3]

# Compile/fit params
batch_size = 80
nb_epoch = 10

lg.info('Loading data...')
# Expect x to be a list of sentences. Y to be a one-hot encoding of the categories.
(xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()

lg.info('Creating vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

lg.info(str(vocab))
test_data = data_helpers.encode_data(x_test, maxlen, vocab, vocab_size, check)

lg.info('Build model...')
model = tweet2vec.model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter)

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
                                                     batch_size=batch_size)

    # accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    lg.info('Epoch: {}'.format(e))

    for x_train, y_train in batches:

        # todo: synonym-replaced texts dataset needed
        f = model.train_on_batch(x_train, x_train)
        loss += f
        loss_avg = loss / step
        # accuracy += f[1]
        # accuracy_avg = accuracy / step

        if step % 100 == 0:
            lg.info('  Step: {}'.format(step))
            # lg.info('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
            lg.info('\tLoss: {}.'.format(loss_avg))
        step += 1

    # test_accuracy = 0.0
    test_loss = 0.0
    test_step = 1
    test_loss_avg = 0.0

    for x_test_batch, y_test_batch in test_batches:
        # todo: synonym-replaced texts dataset needed
        f_ev = model.test_on_batch(x_test_batch, x_test_batch)
        test_loss += f_ev  # [0]
        test_loss_avg = test_loss / test_step
        # test_accuracy += f_ev[1]
        # test_accuracy_avg = test_accuracy / test_step
        test_step += 1

    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    lg.info('Epoch {}. Loss: {}.\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, e_elap, t_elap))

if save:
    lg.info('Saving model params...')
    json_string = model.to_json()

    with open(model_name_path, 'w') as f:
        json.dump(json_string, f, ensure_ascii=False)

    model.save_weights(model_weights_path, overwrite=True)
