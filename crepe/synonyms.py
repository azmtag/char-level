# coding=utf-8

import random
import codecs

from collections import defaultdict

#===================================#

class Synonyms(object):

    #-----------------------------------#

    def __init__(self, synonyms_path='data/synonyms', freq_path='data/freq'):
        self.__synonyms = self.__read_dict(synonyms_path, lambda x: [y.strip() for y in x.split(",")])
        self.__freq = self.__read_dict(freq_path, lambda x: float(x))

        self.__validate_data()

    #-----------------------------------#

    # validate that each word in the synonyms data set have a corresponding frequency
    def __validate_data(self):
        for word, values in self.__synonyms.items():
            assert word in self.__freq

            for value in values:
                assert value in self.__freq

    #-----------------------------------#

    def synonyms(self, word):
        return self.__synonyms[word]

    #-----------------------------------#

    def frequency(self, word):
        return self.__freq[word]

    #-----------------------------------#

    def has(self, word):
        return word in self.__synonyms

    #-----------------------------------#

    def replace(self, word, skip_original_word = False ):
        to_calc = [(x, self.__freq[x]) for x in self.__synonyms[word]]

        if not skip_original_word:
           to_calc.append((word, self.__freq[word]))

        if len(to_calc):
            return self.choose(to_calc)

        return word

    #-----------------------------------#

    def __read_dict(self, path_name, func_proc_values = None):
        result_dict = {}

        with codecs.open(path_name, 'r', 'utf-8') as fin:
            for line in fin:
                tokens = line.split("#")

                if func_proc_values is None:
                    result_dict[tokens[0]] = tokens[1]
                else:
                    result_dict[tokens[0]] = func_proc_values(tokens[1])

        return result_dict

    #-----------------------------------#

    # take list of tuples (word, freq)
    @staticmethod
    def choose(pairs):
        all = 0.0

        for pair in pairs:
            word, freq = pair
            all += freq

        p = random.uniform(0.0, 100.0)

        offset = 0.0
        inx = 0
        for pair in pairs:
            word, freq = pair
            norm_freq = (freq / all) * 100.0

            if p > offset and p < offset + norm_freq:
                return word

            offset += norm_freq
            inx += 1

        first_pair = pairs[0]
        word, _ = first_pair
        return word

#===================================#

if __name__ == "__main__":
    TESTS = False

    if TESTS:
        #===================================#
        # Check generator
        #===================================#
        def validate_pairs(pairs):
            print("Validate pairs: ", str(pairs))
            eps = 0.05
            stat = defaultdict(lambda: 0)
            N = 100000

            for _ in xrange(N):
                stat[Synonyms.choose(pairs)] += 1

            sum = 0.0
            for pair in pairs:
                word, freq = pair
                sum += freq

            # validate distribution
            for pair in pairs:
                word, freq = pair

                freq_norm = freq / sum
                stat_norm = (stat[word] + 0.0) / N

                print("assert on equal : %f(orig_dist) == %f(stat_dist)?" % (freq_norm, stat_norm))

                assert( abs(freq_norm - stat_norm) < eps)

        #-----------------------------------#

        validate_pairs(pairs = [('a', 15), ('b', 9), ('c', 1)])
        validate_pairs(pairs = [('a', 1000), ('b', 1)])
        validate_pairs(pairs = [('a', 7.2), ('b', 3.1), ('c', 9.9), ('d', 1.5), ('f', 42.8)])

        #===================================#
        # Check
        #===================================#

        model = Synonyms()
        word = u'регулировать'
        assert model.has(word) == True

        data1 = [(syn, model.frequency(syn)) for syn in model.synonyms(word)]
        data2 = data1 + [(word, model.frequency(word))]
        validate_pairs(data1)
        validate_pairs(data2)

        model.replace(word)

