import csv
import math
from sklearn.model_selection import train_test_split

def read_file(filename):
    positive = []
    negative = []

    with open(filename, encoding="utf-8") as file:
        f = csv.reader(file, delimiter="\t")
        all_data = []

        for line in f:
            if len(line) >= 2:
                all_data.append(line)

        train, test = train_test_split(all_data, test_size=0.2)

        for line in train:
            if(len(line) >= 2):
                if line[0] == '1':
                    positive.append(line[1])
                if line[0] == '0':
                    negative.append(line[1])



    offsets = [(len(positive) / (len(positive) + len(negative))), (len(negative) / (len(positive) + len(negative)))]

    return positive, negative, test, offsets


def get_priors(positive, negative, min_count = 0):
    total_words = 0
    pos_count = {}
    neg_count = {}

    for line in positive:
        line = line.split(" ")
        for word in line:
            total_words += 1
            if word not in pos_count.keys():
                pos_count[word] = 1
            else:
                pos_count.update({word: pos_count[word]+1})

    for line in negative:
        line = line.split(" ")
        for word in line:
            total_words += 1
            if word not in neg_count.keys():
                neg_count[word] = 1
            else:
                neg_count.update({word: neg_count[word]+1})

    neg_priors = {}
    pos_priors = {}

    for key in neg_count.keys():
        if neg_count[key] >= min_count:
            neg_priors[key] = math.log(neg_count[key]/ total_words)

    for key in pos_count.keys():
        if pos_count[key] >= min_count:
            pos_priors[key] = math.log(pos_count[key] / total_words)

    return pos_priors, neg_priors

# def get_priors(data, min_count=0, max_iters=0):
#     counts = get_counts(data, max_iters)
#     priors = {}
#
#     clean_counts = clean_data(counts, min_count)
#
#     total_words = 0
#
#     for key in counts.keys():
#         total_words += counts[key]
#
#     for key in clean_counts.keys():
#         priors[key] = math.log(clean_counts[key]/total_words)
#
#     return priors, total_words


def clean_data(data, min_count):
    fake_data = data.copy()

    for entry in fake_data.keys():
        if data[entry] <= min_count:
            del data[entry]

    # if '@USER' in data.keys():
    #     del data['@USER']

    return data
def get_counts(data, max_iters):
    counts = {}

    for line in data:
        tripper = 0
        iters = 0
        for word in line.split(" "):
            # if word.lower == "not" or word.lower == "nodyn":
            #     tripper = 1
            #
            # if tripper == 1:
            #     if iters <= max_iters:
            #         word = word + '*'
            #         iters += 1
            #     else:
            #         tripper = 0

            if word in counts.keys():
                counts.update({word: counts[word]+1})
            else:
                counts[word] = 1

    return counts