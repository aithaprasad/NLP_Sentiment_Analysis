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

def get_priors(data):
    counts = get_counts(data)
    priors = {}

    total_words = 0

    for key in counts.keys():
        total_words += counts[key]

    for key in counts.keys():
        priors[key] = (counts[key]/total_words)

    return priors, total_words

def get_counts(data):
    counts = {}

    for line in data:
        for word in line.split(" "):
            if word in counts.keys():
                counts.update({word: counts[word]+1})
            else:
                counts[word] = 1

    return counts