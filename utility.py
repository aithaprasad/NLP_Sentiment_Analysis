import csv
import math
from sklearn.model_selection import train_test_split


# Reads the file and returns lists of positive and negative sentiment
def read_file(filename):
    positive = []
    negative = []

    # Loading data from file
    with open(filename, encoding="utf-8") as file:
        f = csv.reader(file, delimiter="\t")
        all_data = []

        for line in f:
            if len(line) >= 2:
                all_data.append(line)

        # Splitting data into train and test before anything else
        train, test = train_test_split(all_data, test_size=0.2)

        # Dividing positive and negative sentiments
        for line in train:
            if(len(line) >= 2):
                if line[0] == '1':
                    positive.append(line[1])
                if line[0] == '0':
                    negative.append(line[1])

    # Calculate base percentage of each class
    offsets = [(len(positive) / (len(positive) + len(negative))), (len(negative) / (len(positive) + len(negative)))]

    return positive, negative, test, offsets


# Calculate the priors and return
def get_priors(positive, negative, min_count = 0, max_iters = 0):
    total_words = 0
    pos_count = {}
    neg_count = {}

    # Get counts of each word in each class, along with total words in all tweets
    pos_count, total_words = get_counts(positive, max_iters)
    neg_count, total_words = get_counts(negative, max_iters, total_words)

    # Call clean data function
    neg_count = clean_data(neg_count, min_count)
    pos_count = clean_data(pos_count, min_count)

    neg_priors = {}
    pos_priors = {}

    # Calculate priors by dividing occurences of word in class divided  by all words, then logged
    for key in neg_count.keys():
        if neg_count[key] >= min_count:
            neg_priors[key] = math.log(neg_count[key]/ total_words)

    # Same as above for other class
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


# Clean data, anything under specified minimum count will be removed
def clean_data(data, min_count):
    fake_data = data.copy()

    for entry in fake_data.keys():
        if data[entry] <= min_count:
            del data[entry]

    # if '@USER' in data.keys():
    #     del data['@USER']

    return data

# Get the counts of words, also includes not hack
def get_counts(data, max_iters, total_count=0):
    counts = {}

    # For every tweet run loop
    for line in data:
        tripper = 0
        iters = 0
        # For every word in tweet, run loop
        for word in line.split(" "):
            # If the word is not or the welsh equivalent, start adding * to following words
            if word.lower == "not" or word.lower == "nodyn":
                tripper = 1

            total_count += 1

            # If previous word was not, add * if not above max iters
            if tripper == 1:
                if iters <= max_iters:
                    word = word + '*'
                    iters += 1
                # If above max iters, stop adding *
                else:
                    tripper = 0

            # Update word in dict
            if word in counts.keys():
                counts.update({word: counts[word]+1})
            else:
                counts[word] = 1

    return counts, total_count