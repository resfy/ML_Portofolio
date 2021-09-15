# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_accuracy') > 0.91 and logs.get('accuracy') > 0.91):
            print("\nReached 91.0% accuracy so cancelling training!")
            self.model.stop_training = True


def solution_B4():
    bbc = pd.read_csv(
        'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/bbc-text.csv')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    train_size = int(len(bbc) * training_portion)
    train_labels, test_labels = train_test_split(bbc['category'], train_size)
    train_text, test_text = train_test_split(bbc['text'], train_size)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                      oov_token=oov_tok,
                                                      char_level=False)

    # fit tokenizer to our training text data
    tokenizer.fit_on_texts(train_text)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(train_text)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(test_text)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # one_hot_train = pd.get_dummies(train_labels)
    # one_hot_test = pd.get_dummies(test_labels)

    # Use sklearn utility to convert label strings to numbered index
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    y_train = encoder.transform(train_labels)
    y_test = encoder.transform(test_labels)
    # one hot encoding
    num_classes = np.max(y_train) + 1
    one_hot_train = tf.keras.utils.to_categorical(y_train, num_classes)
    one_hot_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(125, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(padded, one_hot_train, epochs=200,
              validation_data=(testing_padded, one_hot_test),
              callbacks=[myCallback()])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.


model = solution_B4()
model.save("model_B4.h5")
