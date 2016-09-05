import json
import pandas
import sklearn
from sklearn.cross_validation import train_test_split

input_file = 'data/restoclub/restoclub.reviews.json'
train_file = 'data/train.csv'
test_file = 'data/test.csv'

def read_file(fname):
    with open(fname) as data_file:
        data = json.load(data_file)
    return data


def count_rating_type(data):
    all_ratings = {}
    for row in data:
        for rating in row['ratings'].keys():
            if rating in all_ratings.keys():
                all_ratings[rating] += 1
            else:
                all_ratings[rating] = 1
    return all_ratings


def to_tuples(data):
    return [(
        str(r['ratings']['total']) if 'total' in r['ratings'].keys() else '0',
        str(r['ratings']['сервис']) if 'сервис' in r['ratings'].keys() else '0',
        str(r['ratings']['интерьер']) if 'интерьер' in r['ratings'].keys() else '0',
        str(r['ratings']['кухня']) if 'кухня' in r['ratings'].keys() else '0',
        r['text'].replace('\n', ' ').replace('\r', '')
    ) for r in data]


if __name__ == '__main__':
    data = read_file(input_file)
    data_tuples = to_tuples(data)
    train, test = train_test_split(data_tuples, test_size=0.3, random_state=42)
    pandas.DataFrame(train).to_csv('train.csv', index=False, header=False)
    pandas.DataFrame(test).to_csv('test.csv', index=False, header=False)
    # print(to_tuples(data[:1000]))






