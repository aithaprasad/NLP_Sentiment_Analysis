import utility
import math


alphas = [0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16]

min_counts = [0, 1, 2, 3, 4, 5]

max_iters = 2

def main():
    print("Loading data...")
    positive, negative, test, offsets = utility.read_file("train.tsv")

    pos_offset = math.log(offsets[0])
    neg_offset = math.log(offsets[1])


    for min_count in min_counts:
        #for max_iter in max_iters:

        print("\n\nCalculating priors for min count of ", min_count, "...", sep="")
        positive_priors, negative_priors = utility.get_priors(positive, negative, min_count, max_iters)

        print("\nPredicting...")
        predict(test, [pos_offset, neg_offset], positive_priors, negative_priors)

        test_priors(positive_priors, negative_priors)


def test_priors(pos, neg):
    total_pos = 0
    total_neg = 0

    for key in pos.keys():
        if key in neg.keys():
            if pos[key] > neg[key]:
                #print("Pos: ", pos[key], "   Neg: ", neg[key], "       KEY: ", key, "     POSITIVE")
                total_pos += 1
            if pos[key] < neg[key]:
                #print("Pos: ", pos[key], "   Neg: ", neg[key], "       KEY: ", key, "     NEGATIVE")
                total_neg += 1

    print("\n\nTotal Positive: ", total_pos, "        Total Negative: ", total_neg)

def predict(test_data, offsets, positive_priors, negative_priors):
    for alpha in alphas:
        all_predicts = []
        total_neg = 0
        total = 0
        for text in test_data:
            pos_sum = offsets[0]
            neg_sum = offsets[1]

            predict = -1

            tripper = 0
            iter = 0

            for word in text[1]:
                if word.lower == "not" or word.lower == "nodyn":
                    tripper = 1

                if tripper == 1:
                    if iter <= max_iters:
                        word = word + "*"
                        iters += 1
                    else:
                        trigger = 0
                if word in positive_priors.keys():
                    pos_sum += positive_priors[word] + math.log(alpha)
                if word in negative_priors.keys():
                    neg_sum += negative_priors[word] + math.log(alpha)

                if word not in positive_priors.keys() and word not in negative_priors.keys():
                    neg_sum += alpha
                    pos_sum += alpha


            if neg_sum > pos_sum:
                predict = 0
                total_neg += 1
            if pos_sum > neg_sum:
                predict = 1

            total += 1
            all_predicts.append([str(predict), text[0]])

        print(total_neg, total)

        print("Calculating scores for Alpha ", alpha, "...", sep="")

        print("\nScores for Alpha ", alpha, ":", sep="")
        print(accuracy(all_predicts), "% Accuracy", sep="")
        print(precision(all_predicts), "% Precision", sep="")
        print(recall(all_predicts), "% Recall", sep="")
        print(fscore(all_predicts), "% F Score", sep="")


def accuracy(all_predicts):
    total_right = 0
    total = 0

    for predict in all_predicts:
        if predict[0] == predict[1]:
            total_right += 1

        total += 1

    acc = 100 * (total_right/total)

    return acc


def precision(all_predicts):
    true_pos = 0
    false_pos = 0

    for predict in all_predicts:
        if predict[0] == '1' and predict[1] == '1':
            true_pos += 1
        if predict[0] == '1' and predict[1] != '1':
            false_pos += 1

    precis = 100 * (true_pos / (true_pos + false_pos))

    return precis

def recall(all_predicts):
    true_pos = 0
    false_neg = 0

    for predict in all_predicts:
        if predict[0] == '1' and predict[1] == '1':
            true_pos += 1
        if predict[0] == '0' and predict[1] == '1':
            false_neg += 1


    reca = 100 * (true_pos / (true_pos + false_neg))

    return reca

def fscore(all_predicts):
    rec = recall(all_predicts)
    prec = precision(all_predicts)

    if prec + rec != 0:
        f_score = ((2 * prec * rec)/(prec + rec))
    else:
        f_score = 0

    return f_score


if __name__ == "__main__":
    main()