# coding:utf-8
"""
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
    Run on GPU: KERAS_BACKEND=tensorflow python3 main.py
"""

from __future__ import division
from __future__ import print_function

import argparse as ap
import cPickle
import datetime
import re

import pymystem3 as ms
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm.classes import SVC

import data_helpers


def my_print(s):
    print("[" + str(datetime.datetime.now()) + "] " + s)


parser = ap.ArgumentParser(description='our baselines')

parser.add_argument('--dataset', type=str,
                    choices=['restoclub', 'okstatus', 'okuser'],
                    default='okstatus',
                    help='default=okstatus, choose dataset')

parser.add_argument('--model', type=str,
                    choices=['all', 'svm', 'logreg', 'gbt'],
                    default='all',
                    help='default=all, choose model')

parser.add_argument('--syns', action="store_true",
                    help='default=False; use synonyms')

parser.add_argument('--pref', type=str, default="default",
                    help='default=None (do not save); prefix for saving models')

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
    models.append(LogisticRegression(C=0.8))

if args.model == "svm" or args.model == "all":
    models.append(SVC(C=0.8))

if args.model == "gbt" or args.model == "all":
    models.append(GradientBoostingClassifier(n_estimators=150, max_depth=15, verbose=True))

if args.model not in ["svm", "all", "logreg", "gbt"]:
    my_print("NO SUCH MODEL: " + args.model)
    raise

print("Training...")

# vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2), max_df=0.9)
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_df=0.9)
vectorizer.fit(xt)

# vectorizing
X_train = vectorizer.transform(xt)
y_train = yt

my_print("X_train, y_train shapes: " + str(X_train.shape) + " " + str(y_train.shape))

X_test = vectorizer.transform(x_test)
y_test = y_test

for model in models:
    print()
    print(model)
    print()
    try:
        model.fit(X_train, y_train)
        my_print("Accuracy: " + str(model.score(X_test, y_test)))
    except:
        model.fit(X_train.toarray(), y_train)
        my_print("Accuracy: " + str(model.score(X_test.toarray(), y_test)))

with open(args.model + "_classifier_" + args.pref + ".pkl", "wb") as fid:
    cPickle.dump(models, fid)
    cPickle.dump(vectorizer, fid)
