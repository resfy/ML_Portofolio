# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_accuracy') > 0.99 and logs.get('accuracy') > 0.98):
            print("\nReached 98.0% accuracy so cancelling training!")
            self.model.stop_training = True


def solution_C2():
    mnist = tf.keras.datasets.mnist.load_data()
    (x_train,y_train),(x_test,y_test)=mnist

    x_train=x_train/255.
    x_test=x_test/255.

    # Reshape input data from (28, 28) to (28, 28, 1)
    w, h = 28, 28
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)
    x_test = x_test.reshape(x_test.shape[0], w, h, 1)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # YOUR CODE HERE
    model = tf.keras.Sequential([
        # tf.keras.layers.Reshape((28,28,1)),
        tf.keras.layers.Conv2D(16, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    model.fit(x_train,
              y_train,
              batch_size=64,
              epochs=10,
              validation_data=(x_test, y_test),
              callbacks=[myCallback()])

    # YOUR CODE HERE

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    if __name__ == '__main__':
        model = solution_C2()
        model.save("model_C2.h5")