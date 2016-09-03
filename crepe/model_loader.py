# coding=utf-8
import argparse as ap
import json

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.models import model_from_json
import os

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

model.compile(optimizer=optimizer, loss=model.loss, metrics=model.metrics)

print("Model loaded and compiled")

# ============= TEST DATA =============

if args.dataset == 'restoclub':
    (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()
else:
    raise Exception("Unknown dataset: " + args.dataset)

# ============= EVAL ==================

# todo: kfold

xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)
test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                 vocab_size, check, model.input.shape[0],
                                                 batch_size=int(args.batch))

scores = model.evaluate(xi_test, yi_test, verbose=1)
print("scores", scores)
