# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:20:11 2020

@author: logiusti
"""

import cv2
import tensorflow as tf
import pandas as pd
import re
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



#useful constant
img_rows, img_cols, img_depth = 128, 128, 3
input_shape = (img_rows, img_cols, img_depth)

batch_size = 128
num_classes = 1
epochs = 12


#load patients' data
patients_data = pd.read_csv(r'patients_data.csv').to_numpy()

#handle train/test indeces
indexes = np.arange(0, len(patients_data), 1)
np.random.shuffle(indexes)
train_indeces = indexes[:int(len(indexes)*.8)]
test_indeces = indexes[int(len(indexes)*.8):]


#pick patients train/test data
patients_data_train = patients_data[train_indeces]
patients_data_test = patients_data[test_indeces]

no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\NoTumor"
tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\Tumor"


no_tumor_images, tumor_images = [], []

# put the non tumor png files into an array
for image in os.listdir(no_tumor_path):
    if re.search(r"png", image):
        img = cv2.imread(no_tumor_path + '\\' + image)
        img = cv2.resize(img, (img_rows, img_rows), interpolation=cv2.INTER_AREA)
        no_tumor_images.append(img)

print("Non tumor data loaded")

# you know that numpy arrays are matti coatti
no_tumor_images = np.array(no_tumor_images)

# put the tumor png files into an array
for image in os.listdir(tumor_path):
    if re.search(r"png", image):
        img = cv2.imread(tumor_path + '\\' + image)
        img = cv2.resize(img, (img_rows, img_rows), interpolation=cv2.INTER_AREA)
        tumor_images.append(img)


print("Tumor data loaded")
#the same as before
tumor_images = np.array(tumor_images)


#dataset is built up by concatenating vabbe' che te lo dico affa'
X = np.concatenate((tumor_images, no_tumor_images)).astype('float64')
X /= 255.0 #rescaling everything in [0, 1] is a best practice for imagage processing


print("X is ok and rescaled")

#One if its a tumor, Zero otherwise. Position matters
y = np.concatenate((np.ones(len(tumor_images)) , np.zeros(len(no_tumor_images))))

print("y is ok")

#This will be useless in a while
X_train = X[train_indeces]
X_test = X[test_indeces]

y_train = y[train_indeces]
y_test = y[test_indeces]

print("Dataset splitted in train+test correctly")

cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 ))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))


cnn.add(Conv2D(64,(3, 3),
                 activation='elu'))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))



cnn.add(Conv2D(64, (3, 3),
                 activation='relu'))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(128, (3, 3),
                 activation='relu'))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))


cnn.add(Flatten())
cnn.add(Dense(4096, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(4096, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(1000, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(num_classes, activation='softmax'))





fnn = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_shape=(patients_data.shape[1],), activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

'''

fnn.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model_hist = fnn.fit(
  patients_data_train,
  y_train,
  validation_data=(patients_data_test, y_test),
  epochs=5,
  batch_size=1)


cnn.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model_hist = cnn.fit(
  X_train,
  y_train,
  validation_data=(X_test, y_test),
  epochs=5,
  batch_size=1)


'''

combinedInput = concatenate([fnn.output, cnn.output])

x = Dense(64, activation="relu")(combinedInput)
x = Dense(num_classes, activation="softmax")(x)



# The final model accepts numerical data on the MLP input and images on the CNN input, outputting a single value
model = Model(inputs=[fnn.input, cnn.input], outputs=x)



checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


opt = Adam(learning_rate = 0.00001, amsgrad=True)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



class_weight = {0: 1.6,
                1: 1.0}

model_hist = model.fit(
  [patients_data_train, X_train],
  y_train,
  validation_data=([patients_data_test, X_test], y_test),
  epochs=5,
  batch_size=1,
  callbacks=[cp_callback],
  class_weight=class_weight)

# Compile the model
# Train the model

score = model.evaluate([patients_data_test, X_test], y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
