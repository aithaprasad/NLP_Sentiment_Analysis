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

    return positive, negative, test

def get_priors(data, min_count):
    counts = get_counts(data)
    priors = {}

    clean_counts = clean_data(counts, min_count)

    total_words = 0

    for key in counts.keys():
        total_words += counts[key]

    for key in clean_counts.keys():
        priors[key] = math.log(clean_counts[key]/total_words)

    return priors, total_words


def clean_data(data, min_count):
    fake_data = data.copy()

    for entry in fake_data.keys():
        if data[entry] <= min_count:
            del data[entry]

    return data
def get_counts(data):
    counts = {}

    for line in data:
        tripper = 0
        iters = 0
        for word in line.split(" "):
            if word.lower == "not":
                tripper = 1

            if tripper == 1:
                if iters < 2:
                    word = word + '*'
                    iters += 1
                else:
                    tripper = 0

            if word in counts.keys():
                counts.update({word: counts[word]+1})
            else:
                counts[word] = 1

    return counts