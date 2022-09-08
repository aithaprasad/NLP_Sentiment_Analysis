from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from sklearn import metrics
import numpy as np
from keras.models import load_model


def run_anything_goes():
    data = pd.read_csv('train.tsv', delimiter="\t", header=None, names=['label', 'sentence'])
    test_data = pd.read_csv('train.tsv', delimiter="\t", header=None, names=['label', 'sentence'])
    data.head()  # just making sure data got read and labelled correctly

    X, y = data['sentence'], data['label']
    X_final_test, y_final_test = test_data['sentence'], test_data['label']


    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.20, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(X_train)

    X_train_tok = tokenizer.texts_to_sequences(X_train)
    X_dev_tok = tokenizer.texts_to_sequences(X_dev)
    X_test_tok = tokenizer.texts_to_sequences(X_test)
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(X)

    X_tok = tokenizer.texts_to_sequences(X)
    # X_dev_tok = tokenizer.texts_to_sequences(X_dev)
    X_final_test_tok = tokenizer.texts_to_sequences(X_final_test)
    X_train_tok[60000]

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 100

    X_train_pad = pad_sequences(X_train_tok, padding='post', maxlen=maxlen)
    X_dev_pad = pad_sequences(X_dev_tok, padding='post', maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_tok, padding='post', maxlen=maxlen)
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 100

    X_pad = pad_sequences(X_tok, padding='post', maxlen=maxlen)
    # X_dev_pad = pad_sequences(X_dev_tok, padding='post', maxlen=maxlen)
    X_final_test_pad = pad_sequences(X_final_test_tok, padding='post', maxlen=maxlen)

    model = load_model("neural_network_with_regularization.h5")

    y_final_test_predicted_labels = model.predict(X_final_test_pad)
    y_final_test_predicted_labels = (y_final_test_predicted_labels > 0.5).astype(int)

    final_test_accuracy = metrics.accuracy_score(y_final_test, y_final_test_predicted_labels)
    final_test_precision = metrics.precision_score(y_final_test, y_final_test_predicted_labels, pos_label=1)
    final_test_recall = metrics.recall_score(y_final_test, y_final_test_predicted_labels, pos_label=1)
    final_test_f1score = metrics.f1_score(y_final_test, y_final_test_predicted_labels, pos_label=1)
    final_test_auc_score = metrics.roc_auc_score(y_final_test, y_final_test_predicted_labels)

    print("Accuracy: ", final_test_accuracy)
    print("Precision: ", final_test_precision)
    print("Recall: ", final_test_recall)
    print("F Score: ", final_test_f1score)
