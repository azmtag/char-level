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
from sklearn.ensemble.forest import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm.classes import SVC

import data_helpers


def my_print(s):
    print("[" + str(datetime.datetime.now()) + "] " + s)


parser = ap.ArgumentParser(description='our baselines')

parser.add_argument('--dataset', type=str,
                    choices=['restoclub', 'okstatus', 'okuser'],
                    default='okuser',
                    help='default=okstatus, choose dataset')

parser.add_argument('--model', type=str,
                    choices=['all', 'rf', 'extratrees', 'linreg', 'gbt'],
                    default='all',
                    help='default=all, choose model')

parser.add_argument('--syns', action="store_true",
                    help='default=False; use synonyms')

parser.add_argument('--pref', type=str, default="default",
                    help='default=default; prefix for saving models')

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
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_data_gender()
    mode = 'binary'
elif args.dataset == 'okuser':
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_user_data_gender()
    xt, x_test = list(xt), list(y_test)
    mode = 'binary'
else:
    raise Exception("Unknown dataset: " + args.dataset)

mystem = ms.Mystem()


def normalize_text(s, use_denumberization=False):
    """
    Преобразование текста перед one-hot-encoding
    :param s: initial text for analyzing
    :param use_denumberization: is False by default,
                                because there are improtant predictor numbers such as short phone numbers
    :return: prepared text
    """
    s = re.sub(u"\s+", u" ", re.sub(u"[^A=-Za-zА-Яа-я0-9 ]", u" ", s))
    s = u" ".join(mystem.lemmatize(s)).lower()

    if use_denumberization:
        return re.sub(u"\d+", u"NUMBER", s).encode("utf-8")
    else:
        return s.encode("utf-8")


my_print('Building model...')
initial = datetime.datetime.now()

# todo: find better params with GridSearch, esp. regularizers weights are important

models = []

if args.model == "linreg" or args.model == "all":
    models.append(LinearRegression(n_jobs=2))

if args.model == "extratrees" or args.model == "all":
    models.append(ExtraTreesRegressor(n_jobs=3))

if args.model == "rf" or args.model == "all":
    models.append(RandomForestRegressor(n_jobs=3))

if args.model == "gbt" or args.model == "all":
    models.append(GradientBoostingRegressor(n_estimators=150, max_depth=15, verbose=True))

if args.model not in ["linreg", "all", "rf", "gbt", "extratrees"]:
    my_print("NO SUCH MODEL: " + args.model)
    raise

my_print("Training...")

print(type(xt), xt.shape)

xt = normalize_text(xt)
x_test = normalize_text(x_test)

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
    model.fit(X_train, y_train)
    try:
        my_print("Accuracy: " + str(model.score(X_test, y_test)) + " " + str(model))
    except:
        my_print("Accuracy: " + str(model.score(X_test.toarray(), y_test)) + " " + str(model))

with open(args.model + "_regr_" + args.pref + ".pkl", "wb") as fid:
    cPickle.dump(models, fid)
    cPickle.dump(vectorizer, fid)
