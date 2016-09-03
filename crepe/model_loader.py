# coding=utf-8
import argparse as ap
import datetime
import json
import os

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import Adam

import data_helpers

parser = ap.ArgumentParser(description='our py_crepe loader')

parser.add_argument('--models-path', type=str, required=True,
                    help='no default')

parser.add_argument('--dataset', type=str,
                    choices=['restoclub'],
                    default='restoclub',
                    help='default=restoclub, apply to which dataset')

parser.add_argument('--pref', type=str, required=True,
                    help='no default')

parser.add_argument('--optimizer', type=str, choices=['adam', 'rmsprop'], default='adam',
                    help='default=adam')

parser.add_argument('--loss', type=str, choices=['mean_squared_error'], default='mean_squared_error',
                    help='default=mean_squared_error')

parser.add_argument('--batch', type=int,
                    default=80,
                    help='default=80; test batch size')

parser.add_argument('--gpu_fraction', type=float,
                    default=0.3,
                    help='default=0.3; GPU fraction, please, use with care')

args = parser.parse_args()

# ====== GPU SESSION =====================

print('GPU fraction ' + str(args.gpu_fraction))


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

# ============= vACOAB =============

print('Creating vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

# ============= MODEL ===============

json_file = open(args.models_path + "/" + args.pref + ".json", 'r')
model_as_json = json.load(json_file, encoding="UTF-8")
json_file.close()

model = model_from_json(model_as_json)
model.load_weights(args.models_path + "/" + args.pref + ".h5")

print (model)

if args.optimizer == 'adam':
    optimizer = Adam()
else:
    optimizer = args.optimizer

print("Chosen optimizer: ", optimizer)

model.compile(optimizer=optimizer, loss=args.loss, metrics=['accuracy'])

print("Model loaded and compiled")

# ============= TEST DATA =============

if args.dataset == 'restoclub':
    (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()
else:
    raise Exception("Unknown dataset: " + args.dataset)

# ============= EVAL ==================

# todo: kfold

print('Input size, maxlen', model.input_shape)

xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)
test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                 vocab_size, check, model.input_shape[1],
                                                 batch_size=int(args.batch))

start = datetime.datetime.now()
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

print('Loss: {}. Accuracy: {}\nTotal time: {}\n'.format(test_loss_avg, test_acc_avg, e_elap))
