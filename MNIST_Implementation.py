import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from FMP import *

"""
    This Model is implemented on the basis of the Research paper on Fractional Max Pooling,
    where the author trains the model without any training set augmentation.
    
    Here the input layer size is 28 * 28
    Architecture : 6 layers of (32nC2 - FMP(1.25)), n = 1,...6
    
    The model is trained for 1 epoch on the MNIST dataset, where it achieved an accuracy of
    0.9540 and a loss of 0.1846.
"""
model = keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)))
model.add(FractionalMaxPooling2D())

model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(FractionalMaxPooling2D())

model.add(Conv2D(96, (3, 3), activation="relu", padding="same"))
model.add(FractionalMaxPooling2D())

model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(FractionalMaxPooling2D())

model.add(Conv2D(160, (3, 3), activation="relu", padding="same"))
model.add(FractionalMaxPooling2D())

model.add(Conv2D(192, (3, 3), activation="relu", padding="same"))
model.add(FractionalMaxPooling2D())

model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.astype(np.float32)
X_train = np.expand_dims(X_train, axis=3)

X_test = X_test.astype(np.float32)
X_test = np.expand_dims(X_test, axis=3)
print(X_test.shape)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

model.compile(loss='sparse_categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))