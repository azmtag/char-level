# coding:utf-8
"""
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
    Run on GPU: KERAS_BACKEND=tensorflow python3 main.py
"""

from __future__ import division
from __future__ import print_function

import argparse as ap
import pickle
import datetime

from sklearn import metrics
from sklearn.ensemble.forest import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.base import LinearRegression

import data_helpers


def my_print(s):
    print("[" + str(datetime.datetime.now()) + "] " + s)


parser = ap.ArgumentParser(description='our baselines')

parser.add_argument('--dataset', type=str,
                    choices=['restoclub', 'okstatus', 'okuser'],
                    default='okstatus',
                    help='default=okstatus, choose dataset')

parser.add_argument('--model', type=str,
                    choices=['all', 'rf', 'extratrees', 'linreg', 'gbt'],
                    default='all',
                    help='default=all, choose model')

parser.add_argument('--syns', action="store_true",
                    help='default=False; use synonyms')

parser.add_argument('--pref', type=str, default="age",
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
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_data_age_normalized()
    mode = 'binary'
elif args.dataset == 'okuser':
    (xt, yt), (x_test, y_test) = data_helpers.load_ok_user_data_age_normalized()
    mode = 'binary'
else:
    raise Exception("Unknown dataset: " + args.dataset)

my_print('Building model...')
initial = datetime.datetime.now()

# todo: find better params with GridSearch, esp. regularizers weights are important
models = []

if args.model == "linreg" or args.model == "all":
    models.append(LinearRegression(n_jobs=2))

if args.model == "extratrees" or args.model == "all":
    models.append(ExtraTreesRegressor(n_jobs=3))

if args.model == "rf" or args.model == "all":
    models.append(RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=3))

if args.model == "gbt" or args.model == "all":
    models.append(GradientBoostingRegressor(n_estimators=150))

if args.model not in ["linreg", "all", "rf", "gbt", "extratrees"]:
    my_print("NO SUCH MODEL: " + args.model)
    raise

my_print(" Vectorization...")

try:
    with open("vectorizer.bin", "rb") as inp:
        vectorizer = pickle.load(inp)
except Exception as e:
    print(e)
    # vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2), max_df=0.9)
    vectorizer = TfidfVectorizer(min_df=40, ngram_range=(1, 2), max_df=0.6)
    vectorizer.fit(xt)

    with open("vectorizer.bin", "wb") as vbin:
        pickle.dump(vectorizer, vbin)

X_train = vectorizer.transform(xt)
y_train = yt

my_print("X_train, y_train shapes: " + str(X_train.shape) + " " + str(y_train.shape))

X_test = vectorizer.transform(x_test)
y_test = y_test

my_print("Vectorization done. Training...")

for model in models:

    print()
    my_print(str(model))
    print()

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        with open(args.model + "_regr_" + args.dataset + "_" + args.pref + ".txt", "w") as of:
            of.write("Native score: " + str(model.score(X_test, y_test)) + "\n")
            of.write("MAE: " + str(metrics.mean_absolute_error(y_pred, y_test)) + "\n")
            of.write("MSE: " + str(metrics.mean_squared_error(y_pred, y_test)) + "\n")
            of.write("R2: " + str(metrics.r2_score(y_pred, y_test)) + "\n")

        my_print("Native score: " + str(model.score(X_test, y_test)))
        my_print("MAE: " + str(metrics.mean_absolute_error(y_pred, y_test)))
        my_print("MSE: " + str(metrics.mean_squared_error(y_pred, y_test)))
        my_print("R2: " + str(metrics.r2_score(y_pred, y_test)))
    except:
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        with open(args.model + "_regr_" + args.dataset + "_" + args.pref + ".txt", "w") as of:
            of.write("Native score: " + str(model.score(X_test, y_test)) + "\n")
            of.write("MAE: " + str(metrics.mean_absolute_error(y_pred, y_test)) + "\n")
            of.write("MSE: " + str(metrics.mean_squared_error(y_pred, y_test)) + "\n")
            of.write("R2: " + str(metrics.r2_score(y_pred, y_test)) + "\n")

        my_print("MAE: " + str(metrics.mean_absolute_error(model.predict(X_test), y_test)))
        my_print("MSE: " + str(metrics.mean_squared_error(model.predict(X_test), y_test)))
        my_print("R2: " + str(metrics.r2_score(y_pred, y_test)))

with open(args.model + "_regr_" + args.dataset + "_" + args.pref + ".pkl", "wb") as fid:
    pickle.dump(models, fid)
    pickle.dump(vectorizer, fid)
