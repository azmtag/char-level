# coding=utf-8
import argparse as ap
import json

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.models import model_from_json
import os

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

# ============= MODEL ===============

json_file = open(args.models_path + "/" + args.pref + ".json", 'r')
model_as_json = json.load(json_file, encoding="UTF-8")
json_file.close()

model = model_from_json(model_as_json)
model.load_weights(args.models_path + "/" + args.pref + ".h5")

# ============= TEST DATA =============

if args.dataset == 'restoclub':
    (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()
else:
    raise Exception("Unknown dataset: " + args.dataset)

xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)

# ============= EVAL ==================

# todo: kfold
scores = model.evaluate(xi_test, yi_test, verbose=1)
print("scores", scores)
