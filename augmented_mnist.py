# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:02:20 2020

@author: logiusti
"""

from __future__ import print_function

import tensorflow
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.models import Model

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_data = pd.read_csv('mnist_train.csv').to_numpy()
x_test_data = pd.read_csv('mnist_test.csv').to_numpy()


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 name='conv2d1'))
cnn.add(Conv2D(64, (3, 3), activation='relu',
                 name='conv2d2'))
cnn.add(MaxPooling2D(pool_size=(2, 2),
                 name='mp1'))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu',
                 name='dense1'))
cnn.add(Dropout(0.5))
cnn.add(Dense(num_classes, activation='softmax',
                 name='dense2'))



fnn = tensorflow.keras.models.Sequential([
  tensorflow.keras.layers.Dense(128, input_shape=(12,),activation='relu', name='pino'),
  tensorflow.keras.layers.Dropout(0.2),
  tensorflow.keras.layers.Dense(10, activation='softmax')
])



combinedInput = concatenate([fnn.output, cnn.output])

x = Dense(4, activation="relu", name='dense3')(combinedInput)
x = Dense(10, activation="softmax", name='dense4')(x)

# The final model accepts numerical data on the MLP input and images on the CNN input, outputting a single value
model = Model(inputs=[fnn.input, cnn.input], outputs=x)



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model_hist = model.fit(
  [x_train_data, x_train],
  y_train,
  validation_data=([x_test_data, x_test], y_test),
  epochs=5,
  batch_size=10)

# Compile the model
# Train the model

score = model.evaluate([x_test_data, x_test], y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])