'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
Run on GPU: KERAS_BACKEND=tensorflow python3 main.py
'''

from __future__ import print_function
from __future__ import division
import json,os
import py_crepe
import datetime
import numpy as np
import data_helpers
import argparse as ap
np.random.seed(123)  # for reproducibility

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def my_print(s):
    print("[" + str(datetime.datetime.now()) + "] " + s)

# for reproducibility
np.random.seed(123)

parser = ap.ArgumentParser(description='our py_crepe')
parser.add_argument('--epochs', type=int,
                    default=50,
                    help='default=50; epochs count')

parser.add_argument('--dataset', type=str,
                    choices=['restoclub', 'ok'],
                    default='restoclub',
                    help='default=restoclub, choose dataset')

parser.add_argument('--maxlen', type=int,
                    default=1024,
                    help='default=1024; max sequence length')

parser.add_argument('--rnn', type=str, choices=['SimpleRNN', 'LSTM', 'GRU'],
                    default='SimpleRNN',
                    help='default=SimpleRNN; recurrent layers type')

parser.add_argument('--rnndim', type=int,
                    default=64,
                    help='default=64; recurrent layers dimensionality')

parser.add_argument('--batch', type=int,
                    default=80,
                    help='default=80; training batch size')

parser.add_argument('--syns', action="store_true", help='default=False; use synonyms')

parser.add_argument('--gpu_fraction', type=float,
                    default=0.2,
                    help='default=0.2; GPU fraction, please, use with care')

parser.add_argument('--pref', type=str, default=None, help='default=None (do not save); prefix for saving models')


args = parser.parse_args()


def get_session(gpu_fraction=args.gpu_fraction):
    """
        Allocating only gpu_fraction of GPU memory for TensorFlow.
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

#Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = args.maxlen
# maxlen = 1014

#Model params
#Filters for conv layers
nb_filter = args.rnndim
#Number of units in the dense layer
dense_outputs = 1024
#Conv layer kernel size
filter_kernels = [7, 7, 3, 3, 3, 3]
#Number of units in the final output layer. Number of classes.
cat_output = 10

#Compile/fit params
batch_size = args.batch
nb_epoch = args.epochs

my_print('Loading dataset %s...' % ( args.dataset + " with synonyms" if args.syns else args.dataset ))
if args.dataset == 'restoclub':
    if args.syns:
        (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data_with_syns()
    else:
        (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()
elif args.dataset == 'ok':
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_data_gender()
else:
    raise Exception("Unknown dataset: " + args.dataset)

my_print('Creating vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

# test_data = data_helpers.encode_data(x_test, maxlen, vocab, vocab_size, check)

my_print('Building model...')

model = py_crepe.model(filter_kernels, dense_outputs, maxlen, vocab_size,
                       nb_filter, cat_output)

my_print('Fitting model...')
initial = datetime.datetime.now()
for e in range(nb_epoch):
    xi, yi = data_helpers.shuffle_matrix(xt, yt)
    xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)
    if subset:
        batches = data_helpers.mini_batch_generator(xi[:subset], yi[:subset],
                                                    vocab, vocab_size, check,
                                                    maxlen,
                                                    batch_size=batch_size)
    else:
        batches = data_helpers.mini_batch_generator(xi, yi, vocab, vocab_size,
                                                    check, maxlen,
                                                    batch_size=batch_size)

    test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                     vocab_size, check, maxlen,
                                                     batch_size=batch_size)

    accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    print('Epoch: {}'.format(e))
    for x_train, y_train in batches:
        f = model.train_on_batch(x_train, y_train)
        loss += f[0]
        loss_avg = loss / step
        accuracy += f[1]
        accuracy_avg = accuracy / step
        if step % 100 == 0:
            print('  Step: {}'.format(step))
            print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
        step += 1

    test_acc = 0.0
    test_loss = 0.0
    test_step = 1
    test_loss_avg = 0.0
    test_acc_avg = 0.0
    for x_test_batch, y_test_batch in test_batches:
        f_ev = model.test_on_batch(x_test_batch, y_test_batch)
        test_loss += f_ev[0]
        test_loss_avg = test_loss / test_step
        test_acc += f_ev[1]
        test_acc_avg = test_acc / test_step
        test_step += 1
    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    my_print('Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, test_acc_avg, e_elap, t_elap))

    if args.pref != None:
        print('Saving model with prefix %s.%02d...' % (args.pref, e))
        model_name_path = '%s.%02d.json' % (args.pref, e)
        model_weights_path = '%s.%02d.h5' % (args.pref, e)
        json_string = model.to_json()
        with open(model_name_path, 'w') as f:
            json.dump(json_string, f)

        model.save_weights(model_weights_path)
