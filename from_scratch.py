import utility
import math


# Global vars for hyperparameter tuning
alphas = [0.3, 0.4, 0.5, 0.25, 0.28, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.0]

min_counts = [0, 1, 2, 3, 4, 5]

max_iters = [0, 1, 2, 3, 4]


# Runs majority of program
def main():
    max_acc = {
        'score': 0,
        'min_count': 0,
        'max_iter': 0,
        'alpha': 0
    }
    max_pre = {
        'score': 0,
        'min_count': 0,
        'max_iter': 0,
        'alpha': 0
    }
    max_rec = {
        'score': 0,
        'min_count': 0,
        'max_iter': 0,
        'alpha': 0
    }
    max_f = {
        'score': 0,
        'min_count': 0,
        'max_iter': 0,
        'alpha': 0
    }

    # Call read_file
    print("Loading data...")
    positive, negative, test, offsets = utility.read_file("train.tsv")

    # Set offsets for easier access
    pos_offset = math.log(offsets[0])
    neg_offset = math.log(offsets[1])

    # Run for every min count in list
    for min_count in min_counts:
        # Run for every max_iter in list
        for max_iter in max_iters:

            # Calculate priors
            print("\n\nCalculating priors for min count of ", min_count, " and max iterations of ", max_iter, "...", sep="")
            positive_priors, negative_priors = utility.get_priors(positive, negative, min_count, max_iters)

            # Tuning for alphas
            for alpha in alphas:
            # Make predictions
                print("\nPredicting...")
                scores = predict(test, [pos_offset, neg_offset], positive_priors, negative_priors, alpha, max_iter)

                if scores[0] > max_acc['score']:
                    max_acc['score'] = scores[0]
                    max_acc['min_count'] = min_count
                    max_acc['max_iter'] = max_iter
                    max_acc['alpha'] = alpha
                if scores[1] > max_pre['score']:
                    max_pre['score'] = scores[1]
                    max_pre['min_count'] = min_count
                    max_pre['max_iter'] = max_iter
                    max_pre['alpha'] = alpha
                if scores[2] > max_rec['score']:
                    max_rec['score'] = scores[2]
                    max_rec['min_count'] = min_count
                    max_rec['max_iter'] = max_iter
                    max_rec['alpha'] = alpha
                if scores[3] > max_f['score']:
                    max_f['score'] = scores[3]
                    max_f['min_count'] = min_count
                    max_f['max_iter'] = max_iter
                    max_f['alpha'] = alpha




    # Helper function
    test_priors(positive_priors, negative_priors)

    print("\n\nMAXIMUM SCORES AND PARAMETERS\n")
    print("ACCURACY: ", max_acc['score'], " MIN COUNT: ", max_acc['min_count'], ' MAX ITERS: ',
          max_acc['max_iter'], ' Alpha: ', max_acc['alpha'])
    print("PRECISION: ", max_pre['score'], " MIN COUNT: ", max_pre['min_count'], ' MAX ITERS: ',
          max_pre['max_iter'], ' Alpha: ', max_pre['alpha'])
    print("RECALL: ", max_rec['score'], " MIN COUNT: ", max_rec['min_count'], ' MAX ITERS: ',
          max_rec['max_iter'], ' Alpha: ', max_rec['alpha'])
    print("F SCORE: ", max_f['score'], " MIN COUNT: ", max_f['min_count'], ' MAX ITERS: ',
          max_f['max_iter'], ' Alpha: ', max_f['alpha'])


# Helper function to better understand how many words are more positive than negative
def test_priors(pos, neg):
    total_pos = 0
    total_neg = 0

    for key in pos.keys():
        # If word is more positive, increment total_pos, else increment total_neg
        if key in neg.keys():
            if pos[key] > neg[key]:
                #print("Pos: ", pos[key], "   Neg: ", neg[key], "       KEY: ", key, "     POSITIVE")
                total_pos += 1
            if pos[key] < neg[key]:
                #print("Pos: ", pos[key], "   Neg: ", neg[key], "       KEY: ", key, "     NEGATIVE")
                total_neg += 1
        else:
            total_pos += 1

    for key in neg.keys():
        if key not in pos.keys():
            total_neg += 1

    # Print total words that are more positive or negative
    print("\n\nTotal Positive: ", total_pos, "        Total Negative: ", total_neg)


# Make predictions based on priors
def predict(test_data, offsets, positive_priors, negative_priors, alpha, m_iter = 0):

    all_predicts = []
    total_neg = 0
    total = 0

    # run for each test tweet
    for text in test_data:
        pos_sum = offsets[0]
        neg_sum = offsets[1]

        predict = -1

        tripper = 0
        iter = 0

        # Split tweet into words
        for word in text[1]:
            # If word is not or welsh equivalent, start adding *
            if word.lower == "not" or word.lower == "nodyn":
                tripper = 1

            # If should be adding stars, do so
            if tripper == 1:
                if iter <= m_iter:
                    word = word + "*"
                    iter += 1
                # If over max iters stop adding *
                else:
                    trigger = 0

            # Calculate the probability of each class
            if word in positive_priors.keys():
                pos_sum += positive_priors[word] - alpha
            if word in negative_priors.keys():
                neg_sum += negative_priors[word] - alpha

            # If word is in neither class, add alpha in place
            if word not in positive_priors.keys() and word not in negative_priors.keys():
                neg_sum -= alpha
                pos_sum -= alpha

        # If neg_sum is greater, predict negative sentiment and vice versa
        if neg_sum > pos_sum:
            predict = 0
            total_neg += 1
        if pos_sum > neg_sum:
            predict = 1

        # Append to predictions in form [PREDICTION, ACTUAL]
        total += 1
        all_predicts.append([str(predict), text[0]])

    # Print total negative predictions vs total predictions (Just for reference)
    print(total_neg, total)

    print("Calculating scores for Alpha ", alpha, "...", sep="")

    scores = [accuracy(all_predicts), precision(all_predicts), recall(all_predicts),fscore(all_predicts)]

    # Call functions to calculate scores and print
    print("\nScores for Alpha ", alpha, ":", sep="")
    print(scores[0], "% Accuracy", sep="")
    print(scores[1], "% Precision", sep="")
    print(scores[2], "% Recall", sep="")
    print(scores[3], "% F Score", sep="")

    return scores

# Calculate the accuracy
def accuracy(all_predicts):
    total_right = 0
    total = 0

    # For each prediction that is right, increment
    for predict in all_predicts:
        if predict[0] == predict[1]:
            total_right += 1

        total += 1

    # Calculate accuracy
    acc = 100 * (total_right/total)

    return acc


# Calculate precision
def precision(all_predicts):
    true_pos = 0
    false_pos = 0

    # Tally true and false positives
    for predict in all_predicts:
        # True positives
        if predict[0] == '1' and predict[1] == '1':
            true_pos += 1
        # False positives
        if predict[0] == '1' and predict[1] != '1':
            false_pos += 1

    # Calculate precision
    precis = 100 * (true_pos / (true_pos + false_pos))

    return precis

# Calculate recall
def recall(all_predicts):
    true_pos = 0
    false_neg = 0

    # Find true positives and false negatives
    for predict in all_predicts:
        # True positive
        if predict[0] == '1' and predict[1] == '1':
            true_pos += 1
        # False negative
        if predict[0] == '0' and predict[1] == '1':
            false_neg += 1

    # Calcualte recall
    reca = 100 * (true_pos / (true_pos + false_neg))

    return reca

# Calculate fscore
def fscore(all_predicts):
    # Get precision and recall
    rec = recall(all_predicts)
    prec = precision(all_predicts)

    # If precision + recall is not 0, calculate f score, otherwise, default to 0
    if prec + rec != 0:
        f_score = ((2 * prec * rec)/(prec + rec))
    else:
        f_score = 0

    return f_score


if __name__ == "__main__":
    main()