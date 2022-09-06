# Load Data
# Clean data
# Get priors

import csv
import math

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
        train, test = train_test_split(all_data)

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


def train_test_split(data, percent = 0.2):
    percent = int(percent * 10)

    train = data[((len(data) // 10) * percent):]
    test = data[:((len(data) // 10) * percent)]

    return train, test

def clean_data(data):
    chars_to_remove = '!@#$%^&*()-_`~=+;:,./<>?\'\"\\|'
    clean_data = []

    for line in data:
        line = ''.join([char for char in line if char not in chars_to_remove])
        line.lower()
        clean_data.append((line))

    return clean_data


def get_counts(pos, neg):
    counts = {}

    for ngram in pos:
        if ngram in counts.keys():
            counts[ngram] = {'pos': (counts[ngram]['pos']+1), 'neg': counts[ngram]['neg']}
        else:
            counts[ngram] = {'pos': 1, 'neg': 0}

    for ngram in neg:
            if ngram in counts.keys():
                counts[ngram] = {'pos': counts[ngram]['pos'], 'neg': (counts[ngram]['neg'] + 1)}
            else:
                counts[ngram] = {'pos': 0, 'neg': 1}

    return counts


def get_priors(counts, pos, neg, alpha):
    total_neg_words = 0
    total_pos_words = 0
    uniques = len(counts.keys())
    priors = {}

    for key in counts.keys():
        total_pos_words += counts[key]['pos']
        total_neg_words += counts[key]['neg']

    for key in counts.keys():
        pos_prior = (counts[key]['pos'] + alpha)/(total_pos_words + (uniques * alpha))
        neg_prior = (counts[key]['neg'] + alpha)/(total_neg_words + (uniques * alpha))

        priors[key] = {'pos': math.log(pos_prior), 'neg':math.log(neg_prior)}

    return priors


def split_data(data):
    sentiment = []
    tweet = []

    for entry in data:
        sentiment.append(entry[0])
        tweet.append(entry[1])

    return sentiment, tweet

def predict(test_data, priors, offsets, n):
    sentiments, data = split_data(test_data)

    data = clean_data(data)

    all_predicts = []

    for line in data:
        predict = -1
        pos_sum = offsets[0]
        neg_sum = offsets[1]
        line = line.split(" ")

        cur_grams = [line[i:i + n] for i in range(len(line) - n + 1)]

        for gram in cur_grams:
            gram = ''.join(gram)
            if gram in priors.keys():
                pos_sum += priors[gram]['pos']
                neg_sum += priors[gram]['neg']

        if pos_sum > neg_sum:
            predict = 1
        else:
            predict = 0

        all_predicts.append(predict)

    acc = calc_acc(all_predicts, sentiments)
    precision = calc_precision(all_predicts, sentiments)
    recall = calc_recall(all_predicts, sentiments)
    fscore = calc_fscore(all_predicts, sentiments)

    scores = [acc, precision, recall, fscore]

    return scores


def calc_acc(predicts, sentiments):
    total_right = 0

    for i in range(0, len(predicts)):
        if str(predicts[i]) == str(sentiments[i]):
            total_right += 1

    acc = 100 * (total_right / len(predicts))

    return acc


def calc_precision(predicts, sentiments):
    true_pos = 0
    false_pos = 0

    for i in range(0, len(predicts)):
        if str(predicts[i]) == '1' and str(sentiments[i]) == '1':
            true_pos += 1
        if str(predicts[i]) == '1' and str(sentiments[i]) != '1':
            false_pos += 1

    precision = 100 * (true_pos / (true_pos + false_pos))

    return precision


def calc_recall(predicts, sentiments):
    true_pos = 0
    false_neg = 0

    for i in range(0, len(predicts)):
        if str(predicts[i]) == '1' and str(sentiments[i]) == '1':
            true_pos += 1
        if str(predicts[i]) == '0' and str(sentiments[i]) == '1':
            false_neg += 1

    recall = 100 * (true_pos / (true_pos + false_neg))

    return recall


def calc_fscore(predicts, sentiments):
    recall = calc_recall(predicts, sentiments)
    precision = calc_precision(predicts, sentiments)
    fscore = 0

    if recall + precision != 0:
        fscore = ((2 * precision * recall) / (precision + recall))

    return fscore


def print_scores(scores):
    print("ACCURACY: ", scores[0], sep="")
    print("PRECISION: ", scores[1], sep="")
    print("RECALL: ", scores[2], sep="")
    print("F SCORE: ", scores[3], sep="")


def min_count_remove(data, min_count):
    for key in data.keys():
        if data[key]['pos'] <= min_count:
            data[key]['pos'] = 0
        if data[key]['neg'] <= min_count:
            data[key]['neg'] = 0

    return data


def ngrams_gen(data, n):
    ngrams = []

    if n > 0:
        for line in data:
            line = line.split(" ")

            cur_grams = [line[i:i+n] for i in range(len(line)-n+1)]

            for gram in cur_grams:
                ngrams.append(''.join(gram))

    return ngrams

def main():
    max_acc = {
        'score': 0,
        'min_count': 0,
        'alpha': 0,
        'ngram': 0
    }
    max_pre = {
        'score': 0,
        'min_count': 0,
        'alpha': 0,
        'ngram': 0
    }
    max_rec = {
        'score': 0,
        'min_count': 0,
        'alpha': 0,
        'ngram': 0
    }
    max_f = {
        'score': 0,
        'min_count': 0,
        'alpha': 0,
        'ngram': 0
    }

    alphas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
    min_counts = [0, 1, 2, 3, 4, 5]
    ngrams = [1, 2, 3, 4]

    pos, neg, test, offsets = read_file('train.tsv')

    pos = clean_data(pos)
    neg = clean_data(neg)

    for ngram in ngrams:
        print("Ngrams of size: ", ngram, "...", sep="")

        pos_ngrams = ngrams_gen(pos, ngram)
        neg_ngrams = ngrams_gen(neg, ngram)

        counts_full = get_counts(pos_ngrams, neg_ngrams)

        for min_count in min_counts:
            counts = min_count_remove(counts_full, min_count)
            print("Min count of ", min_count, "...", sep="")
            for alpha in alphas:
                print("\n\nCalculating priors and predicting for alpha ", alpha, "...\n", sep="")
                priors = get_priors(counts, pos, neg, alpha)

                scores = predict(test, priors, offsets, ngram)

                print_scores(scores)

                if scores[0] > max_acc['score']:
                    max_acc['score'] = scores[0]
                    max_acc['min_count'] = min_count
                    max_acc['alpha'] = alpha
                    max_acc['ngram'] = ngram
                if scores[1] > max_pre['score']:
                    max_pre['score'] = scores[1]
                    max_pre['min_count'] = min_count
                    max_pre['alpha'] = alpha
                    max_pre['ngram'] = ngram
                if scores[2] > max_rec['score']:
                    max_rec['score'] = scores[2]
                    max_rec['min_count'] = min_count
                    max_rec['alpha'] = alpha
                    max_rec['ngram'] = ngram
                if scores[3] > max_f['score']:
                    max_f['score'] = scores[3]
                    max_f['min_count'] = min_count
                    max_f['alpha'] = alpha
                    max_f['ngram'] = ngram

    print("\n\nMAXIMUM SCORES AND PARAMETERS\n")
    print("ACCURACY: ", max_acc['score'], " MIN COUNT: ", max_acc['min_count'], ' Alpha: ', max_acc['alpha'], ' NGrams: ', max_acc['ngram'])
    print("PRECISION: ", max_pre['score'], " MIN COUNT: ", max_pre['min_count'], ' Alpha: ', max_pre['alpha'], ' NGrams: ', max_pre['ngram'])
    print("RECALL: ", max_rec['score'], " MIN COUNT: ", max_rec['min_count'], ' Alpha: ', max_rec['alpha'], ' NGrams: ', max_rec['ngram'])
    print("F SCORE: ", max_f['score'], " MIN COUNT: ", max_f['min_count'], ' Alpha: ', max_f['alpha'], ' NGrams: ', max_f['ngram'])


if __name__ == '__main__':
    main()