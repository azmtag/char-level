# coding:utf-8

import re
from functools import lru_cache

import pandas as pd
from pymystem3.mystem import Mystem

mystem = Mystem()


@lru_cache(maxsize=15000)
def lemmatize_line(w):
    lemmatized = mystem.lemmatize(w)
    return " ".join(lemmatized).strip()


def normalize_text(s, use_denumberization=False):
    """
    Преобразование текста перед one-hot-encoding
    :param s: initial text for analyzing
    :param use_denumberization: is False by default,
                                because there are improtant predictor numbers such as short phone numbers
    :return: prepared text
    """
    s = re.sub(u"\s+", u" ", re.sub(u"[^A-Za-zА-Яа-я0-9 ]+", u" ", s))
    s = u" ".join(map(lambda x: lemmatize_line(x), s.split(" "))).lower()

    if use_denumberization:
        return re.sub(u"\d+", u"NUMBER", s)
    else:
        return s


def collect(df, index, f):
    line_len = len(list(df.ix[:, index]))
    i = 0
    collector = []
    texts = list(df.ix[:, index])
    df.drop(df.columns[[index]], axis=1, inplace=True)

    for line in texts:
        collector.append(normalize_text(line))
        if i % 1000 == 0:
            print(f, i, "/", line_len)
        i += 1
    return collector


def norm_file(fromm, too, target_index):
    with open(too, "w+") as outp:
        with open(fromm) as f:

            print("Data read", f)
            count = 0

            for line in f:
                if count % 1000 == 0:
                    print(count, f)

                splitted = line.split(",")
                pref = ",".join(splitted[:target_index])
                norm_text = normalize_text(" ".join(splitted[target_index:]))
                outp.write(pref)
                outp.write(",")
                outp.write(norm_text)
                outp.write("\n")
                count += 1


norm_file("data/ok/ok_test.csv", "data/ok/ok_test_normalized.csv", 5)
norm_file("data/ok/ok_user_test.csv", "data/ok/ok_user_test_normalized.csv", 4)
norm_file("data/ok/ok_user_train.csv", "data/ok/ok_user_train_normalized.csv", 4)
norm_file("data/ok/ok_train.csv", "data/ok/ok_train_normalized.csv", 5)

