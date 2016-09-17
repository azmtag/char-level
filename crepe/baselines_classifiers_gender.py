# coding:utf-8
"""
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
    Run on GPU: KERAS_BACKEND=tensorflow python3 main.py
"""

from __future__ import division
from __future__ import print_function
from scipy import sparse
import argparse as ap
import datetime
import pickle
import scipy as sp
import data_helpers
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm.classes import SVC


def my_print(s):
    print("[" + str(datetime.datetime.now()) + "] " + s)


parser = ap.ArgumentParser(description='our baselines')

parser.add_argument('--dataset', type=str,
                    choices=['restoclub', 'okstatus', 'okuser'],
                    default='okuser',
                    help='default=okstatus, choose dataset')

parser.add_argument('--model', type=str,
                    choices=['all', 'svm', 'logreg', 'gbt'],
                    default='all',
                    help='default=all, choose model')

parser.add_argument('--syns', action="store_true",
                    help='default=False; use synonyms')

parser.add_argument('--pref', type=str, default="gender",
                    help='default=None (do not save); prefix for saving models')

parser.add_argument('--usepycrepe', type=bool,
                    default=True,
                    help="defaul=False, using crepe features")

args = parser.parse_args()

# set parameters:
subset = None

my_print('Loading dataset %s...' % (args.dataset + " with synonyms" if args.syns else args.dataset))
mode = 'binary'

if args.dataset == 'restoclub':
    if args.syns:
        (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data_with_syns()
    else:
        (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()
    mode = '1mse'
elif args.dataset == 'okstatus':
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_data_gender_normalized()
    mode = 'binary'
elif args.dataset == 'okuser':
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_user_data_gender_normalized()
    mode = 'binary'
else:
    raise Exception("Unknown dataset: " + args.dataset)

my_print('Building model...')
initial = datetime.datetime.now()

# todo: find better params with GridSearch, esp. regularizers weights are important

models = []

if args.model == "logreg" or args.model == "all":
    models.append(LogisticRegression(C=0.8, n_jobs=6))

if args.model == "svm" or args.model == "all":
    models.append(SVC(C=0.8))

if args.model == "gbt" or args.model == "all":
    models.append(GradientBoostingClassifier(n_estimators=150))

# if args.model == "rf" or args.model == "all":
#     models.append(RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=3))

if args.model not in ["svm", "all", "logreg", "gbt"]:
    my_print("NO SUCH MODEL: " + args.model)
    raise

print("Vectorizing...")

try:
    with open(args.dataset + "_vectorizer.bin", "rb") as inp:
        vectorizer = pickle.load(inp)
except Exception as e:
    print(e)
    # vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2), max_df=0.9)
    vectorizer = TfidfVectorizer(min_df=100, ngram_range=(1, 2), max_df=0.3,
                                 token_pattern=r"(?u)\b[А-Яа-я0-9][А-Яа-я0-9]+\b", max_features=40000)
    vectorizer.fit(xt)

    with open(args.dataset + "_vectorizer.bin", "wb") as vbin:
        pickle.dump(vectorizer, vbin)

# vectorizing
X_train = vectorizer.transform(xt)
y_train = yt

my_print("X_train, y_train shapes: " + str(X_train.shape) + " " + str(y_train.shape))

X_test = vectorizer.transform(x_test)
y_test = y_test

if args.usepycrepe:

    np.random.seed(42)

    # http://stackoverflow.com/a/8505754
    train_idx = np.arange(X_train.shape[0])
    np.random.shuffle(train_idx)
    train_idx = train_idx[:X_train.shape[0] // 5]
    train_idx.sort()

    test_idx = np.arange(X_test.shape[0])
    np.random.shuffle(test_idx)
    test_idx = test_idx[:X_test.shape[0] // 5]
    test_idx.sort()

    print("idx ", train_idx.shape, test_idx.shape)

    (pc_train_x, pc_train_y), (pc_test_x, pc_test_y) = \
        data_helpers.load_pycrepe_features_ok_user_gender(train_idx, test_idx)

    print("Selection shapes", pc_train_x.shape, pc_train_y.shape, pc_test_x.shape, pc_test_y.shape)

    X_train = X_train[train_idx, :]
    y_train = y_train[train_idx]
    X_test = X_test[test_idx, :]
    y_test = y_test[test_idx]

    print("Updated shapes", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if X_train.shape[0] != pc_train_x.shape[0]:
        raise Exception("Different shapes " + str(X_train.shape) + " " + str(pc_train_x.shape))

    X_train = sp.sparse.hstack([X_train, pc_train_x])
    X_test = sp.sparse.hstack([X_test, pc_test_x])

    print("Updated after concat shapes", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

for model in models:
    print()
    print(model)
    print()
    try:
        model.fit(X_train, y_train)
        with open(args.model + "_classifier_" + args.dataset + "_" + args.pref + ".txt", "w") as of:
            of.write("Accuracy: " + str(model.score(X_test, y_test)))
        my_print("Accuracy: " + str(model.score(X_test, y_test)))
    except:
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        model.fit(X_train, y_train)
        with open(args.model + "_classifier_" + args.dataset + "_" + args.pref + ".txt", "w") as of:
            of.write("Accuracy: " + str(model.score(X_test, y_test)))
        my_print("Accuracy: " + str(model.score(X_test, y_test)))

with open(args.model + "_classifier_" + args.dataset + "_" + args.pref + ".pkl", "wb") as fid:
    pickle.dump(models, fid)
    pickle.dump(vectorizer, fid)
