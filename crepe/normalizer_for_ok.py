# coding:utf-8

import pandas as pd
import re
import pymystem3 as ms
from pymystem3.mystem import Mystem

mystem = Mystem()


def normalize_text(s, use_denumberization=False):
    """
    Преобразование текста перед one-hot-encoding
    :param s: initial text for analyzing
    :param use_denumberization: is False by default,
                                because there are improtant predictor numbers such as short phone numbers
    :return: prepared text
    """
    s = re.sub(u"\s+", u" ", re.sub(u"[^A-Za-zА-Яа-я0-9 ]+", u" ", s))
    s = u" ".join(mystem.lemmatize(s)).lower()

    if use_denumberization:
        return re.sub(u"\d+", u"NUMBER", s).encode("utf-8")
    else:
        return s.encode("utf-8")


with open("data/ok/ok_user_train.csv") as f:
    df = pd.read_csv(f)
    print("Data read", f)
    df.ix[:, 4] = df.ix[:, 4].map(lambda x: normalize_text(unicode(x.decode("utf-8"))))
    df.to_csv("data/ok/ok_user_train_normalized.csv")


with open("data/ok/ok_user_test.csv") as f:
    df = pd.read_csv(f)
    print("Data read", f)
    df.ix[:, 4] = df.ix[:, 4].map(lambda x: normalize_text(unicode(x.decode("utf-8"))))
    df.to_csv("data/ok/ok_user_test_normalized.csv")


with open("data/ok/ok_train.csv") as f:
    df = pd.read_csv(f)
    print("Data read", f)
    df.ix[:, 4] = df.ix[:, 4].map(lambda x: normalize_text(unicode(x.decode("utf-8"))))
    df.to_csv("data/ok/ok_train_normalized.csv")


with open("data/ok/ok_test.csv") as f:
    df = pd.read_csv(f)
    print("Data read", f)
    df.ix[:, 4] = df.ix[:, 4].map(lambda x: normalize_text(unicode(x.decode("utf-8"))))
    df.to_csv("data/ok/ok_test_normalized.csv")
