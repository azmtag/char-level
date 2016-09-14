# coding=utf-8
"""
Model loader and tested. Usage example
$ KERAS_BACKEND=tensorflow python3 model_loader.py --pref pycrepe.64 --models-path /home/snikolenko/soft/char-level/crepe/models_senti/

"""
import argparse as ap
import datetime
import glob
import json
import os

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
                    default=0.15,
                    help='default=0.15; GPU fraction, please, use with care')

args = parser.parse_args()


def my_print(s):
    print("[" + str(datetime.datetime.now()) + "] " + s)


# ====== GPU SESSION =====================

my_print('GPU fraction ' + str(args.gpu_fraction))
my_print('models path %s' % args.models_path)
my_print('dataset %s' % args.dataset)
my_print('config %s' % args.pref)


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


def get_metrics(model, test_batches):
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
    return test_loss_avg, test_acc_avg, e_elap
    # my_print('Loss: {}\nAccuracy: {}\nTotal time: {}\n'.format(test_loss_avg, test_acc_avg, e_elap))


# KTF.set_session(get_session())

# ============= VOCAB =============

vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

# ============= TEST DATA =============

if args.dataset == 'restoclub':
    (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()
else:
    raise Exception("Unknown dataset: " + args.dataset)

# ============= EVAL ==================

xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)
test_batches = None
my_print("Dataset %s loaded" % args.dataset)
# ============= MODEL ===============

my_print("Loading models from %s..." % (args.models_path + "/" + args.pref + ".*"))

fnames_json = sorted(glob.glob(args.models_path + '/' + args.pref + '*json'))

my_print("\tcompiling for the first time from %s..." % fnames_json[0])
json_file = open(fnames_json[0], 'r')
model_as_json = json.load(json_file, encoding="UTF-8")
json_file.close()

if args.optimizer == 'adam':
    optimizer = Adam()
else:
    optimizer = args.optimizer

model = model_from_json(model_as_json)
model.compile(optimizer=optimizer, loss=args.loss, metrics=['accuracy'])

my_print("\tcompiled! now running with weights loaded from h5 files...")
for fname_json in fnames_json:
    model.load_weights(fname_json[:-4] + "h5")
    # my_print("Chosen optimizer: ", optimizer, "compiling now...")

    test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                     vocab_size, check, model.input_shape[1],
                                                     batch_size=int(args.batch))

    # my_print("Model loaded and compiled.")
    test_loss_avg, test_acc_avg, e_elap = get_metrics(model, test_batches)
    print("%s\t%.8f\t%.8f" % (fname_json.split('/')[-1], test_loss_avg, test_acc_avg))
