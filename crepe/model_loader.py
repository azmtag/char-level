# coding=utf-8
import argparse as ap

from keras.models import model_from_json

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

args = parser.parse_args()

json_file = open(args.models_path + "/" + args.pref + ".json", 'r')
model_as_json = json_file.read()
json_file.close()

model = model_from_json(model_as_json)
model.load_weights(args.models_path + "/" + args.pref + ".h5")

if args.dataset == 'restoclub':
    (xt, yt), (x_test, y_test) = data_helpers.load_restoclub_data()
else:
    raise Exception("Unknown dataset: " + args.dataset)

xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)

scores = model.evaluate(xi_test, yi_test, verbose=1)
print("scores", scores)
