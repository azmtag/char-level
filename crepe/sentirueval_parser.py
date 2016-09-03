# coding=utf-8

import os
from lxml import etree

import pandas as pd


class SentiParseTarget(object):
    def __init__(self, mapper_to_numbers):
        self.results = []
        self.intext = False
        self.current_review_id = None
        self.current_text = ""
        self.current_sentiment = None
        self.mapper_to_numbers = mapper_to_numbers

    def start(self, tag, attrib):
        if tag == 'review':
            self.current_review_id = attrib['id']
        elif tag == 'text':
            self.intext = True
        elif tag == 'category' and attrib['name'] == 'Whole':
            self.current_sentiment = attrib['sentiment']

    def end(self, tag):
        if tag == 'text':
            self.intext = False
        elif tag == 'review':
            self.results.append((self.mapper_to_numbers(self.current_sentiment),
                                 self.current_review_id,
                                 self.current_sentiment,
                                 self.mapper_to_numbers(self.current_sentiment) * 5,
                                 self.current_text.encode("utf-8")))
            self.current_review_id = None
            self.current_sentiment = None
            self.current_text = ""

    def data(self, data):
        if self.intext:
            self.current_text = self.current_text + data

    def comment(self, text):
        pass

    def close(self):
        pass


def num_mapper(sentitag):
    if sentitag == 'positive':
        return 1.0
    elif sentitag == 'negative':
        return 0.0
    elif sentitag == 'both' or sentitag == 'absence' or sentitag == 'neutral':
        return 0.5
    else:
        raise Exception("Shitty parsing? " + sentitag)


targ = SentiParseTarget(mapper_to_numbers=num_mapper)
parser = etree.XMLParser(target=targ)
dir_with_xml = "data/sentirueval/"

for infile_path in [f for f in os.listdir(dir_with_xml) if f.endswith('.xml')]:
    with open(dir_with_xml + infile_path) as infile:
        result = etree.XML(infile.read(), parser)
        dataset = pd.DataFrame(targ.results, columns=['id', 'sentitag', 'val_0-1', 'val_0-5', 'text'])
        with open(str(infile.name) + ".csv", "wb+") as wfile:
            dataset.to_csv(wfile, index=False)

for tipe in ['train', 'test']:
    with open(dir_with_xml + tipe + ".csv", "wb+") as wfile:
        for infile_path in [f for f in os.listdir(dir_with_xml) if f.endswith(tipe + '.xml.csv')]:
            with open(dir_with_xml + infile_path) as infile:
                data = infile.read()
            wfile.write(data)
            wfile.write("\n")
