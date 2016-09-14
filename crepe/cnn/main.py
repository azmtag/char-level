"""

"""

import numpy as np

import cnn_model
import data_helpers
from w2v import train_word2vec
import datetime
import json
import argparse as ap

np.random.seed(2)


def my_print(s):
    print("[" + str(datetime.datetime.now()) + "] " + s)


parser = ap.ArgumentParser(description='our py_crepe')
parser.add_argument('--epochs', type=int,
                    default=50,
                    help='default=50; epochs count')

parser.add_argument('--dataset', type=str,
                    choices=['restoclub', 'okstatus', 'okuser'],
                    default='okstatus',
                    help='default=restoclub, choose dataset')

parser.add_argument('--maxlen', type=int,
                    default=1024,
                    help='default=1024; max sequence length')

parser.add_argument('--rnn', type=str, choices=['SimpleRNN', 'LSTM', 'GRU'],
                    default='SimpleRNN',
                    help='default=SimpleRNN; recurrent layers type')

parser.add_argument('--optimizer', type=str, choices=['adam', 'adagrad', 'rmsprop', 'adadelta'],
                    default='adam',
                    help='default=adam; keras optimizer')

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

parser.add_argument('--pref', type=str, default=None,
                    help='default=None (do not save); prefix for saving models')

parser.add_argument('--variation', type=str, default='CNN-static', help='default=CNN-static')

args = parser.parse_args()

model_variation = args.variation

print('Model variation is %s' % model_variation)

# Model Hyperparameters
sequence_length = 56
embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

# Training parameters
batch_size = 32
num_epochs = 100
val_split = 0.1

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count                        
context = 10  # Context window size

# Load data
print("Loading data...")

# Compile/fit params
batch_size = args.batch
nb_epoch = args.epochs

print('Loading dataset %s...' % (args.dataset + " with synonyms" if args.syns else args.dataset))
mode = 'binary'

(xt, yt), (x_test, y_test) = (None, None), (None, None)

if args.dataset == 'restoclub':
    if args.syns:
        (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data_with_syns()
    else:
        (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()
    mode = '1mse'
elif args.dataset == 'okstatus':
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_data_gender()
    mode = 'binary'
elif args.dataset == 'okuser':
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_user_data_gender()
    mode = 'binary'
else:
    raise Exception("Unknown dataset: " + args.dataset)

my_print('Creating vocab...')

_, vocabulary, vocabulary_inv = data_helpers.build_word_level_data((xt, yt), (x_test, y_test))
# vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

# test_data = data_helpers.encode_data(x_test, maxlen, vocab, vocab_size, check)

my_print('Building model...')

# print(x, y, vocabulary, vocabulary_inv)

if model_variation == 'CNN-non-static' or model_variation == 'CNN-static':
    embedding_weights = train_word2vec(xt, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation == 'CNN-static':
        x = embedding_weights[0][xt]
elif model_variation == 'CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices].argmax(axis=1)

print("Vocabulary Size: {:d}".format(len(vocabulary)))

model = cnn_model.build_model(sequence_length, embedding_dim,
                              filter_sizes, num_filters,
                              dropout_prob, hidden_dims, model_variation,
                              vocabulary, embedding_weights, mode)
print ("Fitting")

# Training model
# ==================================================
# model.fit(x_shuffled,
#           y_shuffled,
#           batch_size=batch_size,
#           nb_epoch=num_epochs,
#           validation_split=val_split,
#           verbose=2)

initial = datetime.datetime.now()

# set parameters:
subset = None

for e in range(nb_epoch):
    xi, yi = data_helpers.shuffle_matrix(xt, yt)
    xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)

    if subset:
        batches = data_helpers.mini_batch_generator(xi[:subset], yi[:subset],
                                                    vocabulary, batch_size=batch_size)
    else:
        batches = data_helpers.mini_batch_generator(xi, yi, vocabulary,
                                                    batch_size=batch_size)

    test_batches = data_helpers.mini_batch_generator(xi_test, yi_test,
                                                     vocabulary, batch_size=batch_size)

    accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    print('Epoch: {} (step-loss-accuracy-timeepoch-timetotal)'.format(e))

    for x_train, y_train in batches:
        f = model.train_on_batch(x_train, y_train)
        loss += f[0]
        loss_avg = loss / step
        accuracy += f[1]
        accuracy_avg = accuracy / step
        if step % 100 == 0:
            print('{}\t{}\t{}\t{}\t{}'.format(step, loss_avg, accuracy_avg, (datetime.datetime.now() - start),
                                              (datetime.datetime.now() - initial)))
        if step % 10000 == 0:
            print('Saving model with prefix %s.%02d.%02dK...' % (args.pref, e, step / 1000))
            model_name_path = '%s.%02d.%02dK.json' % (args.pref, e, step / 1000)
            model_weights_path = '%s.%02d.%02dK.h5' % (args.pref, e, step / 1000)
            json_string = model.to_json()
            with open(model_name_path, 'w') as f:
                json.dump(json_string, f)
            model.save_weights(model_weights_path)
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

    print(
        'Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, test_acc_avg,
                                                                                    e_elap,
                                                                                    t_elap))

    if args.pref is not None:
        print('Saving model with prefix %s.%02d...' % (args.pref, e))
        model_name_path = '%s.%02d.json' % (args.pref, e)
        model_weights_path = '%s.%02d.h5' % (args.pref, e)
        json_string = model.to_json()
        with open(model_name_path, 'w') as f:
            json.dump(json_string, f)

        model.save_weights(model_weights_path, overwrite=True)
