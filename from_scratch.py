import utility
import math


alphas = [0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12]

def main():
    print("Loading data...")
    positive, negative, test = utility.read_file("train.tsv")

    pos_offset = math.log(len(positive)/(len(positive) + len(negative)))
    neg_offset = math.log(len(negative) / (len(positive) + len(negative)))


    print("Calculating priors...")
    positive_priors, pos_count = utility.get_priors(positive)
    negative_priors, neg_count = utility.get_priors((negative))

    print("Predicting...")
    predict(test, [pos_offset, neg_offset], positive_priors, negative_priors)


def predict(test_data, offsets, positive_priors, negative_priors):
    for alpha in alphas:
        all_predicts = []
        for text in test_data:
            pos_sum = offsets[0]
            neg_sum = offsets[1]

            predict = -1

            for word in text[1]:
                if word in positive_priors.keys():
                    pos_sum += positive_priors[word] + alpha
                else:
                    pos_sum += alpha

                if word in negative_priors.keys():
                    neg_sum += negative_priors[word] + alpha
                # else:
                    # neg_sum += alpha

            if neg_sum > pos_sum:
                predict = 0
            else:
                predict = 1

            all_predicts.append([str(predict), text[0]])

        print("Calculating accuracy for Alpha ", alpha, "...", sep="")

        accuracy(all_predicts)


def accuracy(all_predicts):
    total_right = 0
    total = 0

    for predict in all_predicts:
        if predict[0] == predict[1]:
            total_right += 1

        total += 1

    print(100 * (total_right/total), '% Accuracy', sep='')

if __name__ == "__main__":
    main()