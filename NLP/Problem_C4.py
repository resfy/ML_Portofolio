# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_accuracy') > 0.80 and logs.get('accuracy') > 0.80):
            print("\nReached 80.0% accuracy so cancelling training!")
            self.model.stop_training = True


def solution_C4():
    data_url = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    sarcasmdict =[]
    # YOUR CODE HERE
    for line in open('sarcasm.json', 'r'):
      sarcasmdict.append(json.loads(line))

    for item in sarcasmdict:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    train_text, test_text = train_test_split(sentences,training_size)
    train_label,test_label = train_test_split(labels,training_size)

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

    training_padded = np.array(padded)
    training_labels = np.array(train_label)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(test_label)

    model = tf.keras.Sequential([
    # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(125, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(training_padded, training_labels, epochs=200,
              validation_data=(testing_padded, testing_labels),
              callbacks=[myCallback()])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_C4()
    model.save("model_C4.h5")