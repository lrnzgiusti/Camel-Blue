# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:27:09 2020

@author: logiusti
"""

# %% Import libraries
import cv2
import tensorflow as tf
import pandas as pd
import re
import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



# %%useful constants
img_rows, img_cols, img_depth = 75, 75, 3
input_shape = (img_rows, img_cols, img_depth)

batch_size = 128
num_classes = 2
epochs = 12


# %% Load Radiomics dataset
radiomics_features_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\radiomics\\"
X_train_tumor = pd.read_csv(radiomics_features_path + "radiomics_train_tumor_feature.csv")
X_train_no_tumor = pd.read_csv(radiomics_features_path + "radiomics_train_no_tumor_feature.csv")

X_valid_tumor = pd.read_csv(radiomics_features_path + "radiomics_valid_tumor_feature.csv")
X_valid_no_tumor = pd.read_csv(radiomics_features_path + "radiomics_valid_no_tumor_feature.csv")

# %% Append both classes and shuffle the dataset. we'll use the image_name to load the image datase
X_train = X_train_tumor.append(X_train_no_tumor, ignore_index=True).sample(frac=1)
X_valid = X_valid_tumor.append(X_valid_no_tumor, ignore_index=True)#.sample(frac=1)

# %% Label goes in a separate variable and the classes are mapped to a boolean number (we can use a boolean vector but it's fine)
y_train = X_train['Label'].to_numpy()
y_valid = X_valid['Label'].to_numpy()



y_train = np.vectorize({'Tumor':1, 'No_Tumor':0}.get)(y_train)
y_valid = np.vectorize({'Tumor':1, 'No_Tumor':0}.get)(y_valid)


y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)

# %% Load the image dataset, we'll use the image name to keep the data aligned

train_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\train"
valid_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\validation"


train_images, valid_images = [], []

for idx,row in X_train.iterrows():

    img = cv2.imread(train_path + '\\' + row['Label'] + '\\' + row['Image_Name'])
    img = cv2.resize(img, (img_rows, img_rows), interpolation=cv2.INTER_AREA)
    train_images.append(img)


print("Training images loaded")

for idx,row in X_valid.iterrows():
    img = cv2.imread(valid_path + '\\' + row['Label'] + '\\' + row['Image_Name'])
    img = cv2.resize(img, (img_rows, img_rows), interpolation=cv2.INTER_AREA)
    valid_images.append(img)


print("Validation images loaded")



# %% Remove the labels and the image name from the X's

del X_train['Label']
del X_train['Image_Name']

del X_valid['Label']
del X_valid['Image_Name']


# %% Convert pandas dataframe and python lists to numpy arrays

X_train = X_train.to_numpy()
X_valid = X_valid.to_numpy()


train_images = np.array(train_images).astype(np.float64)
valid_images = np.array(valid_images).astype(np.float64)

# %% Scale Data & images

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)

train_images /= 255.0
valid_images /= 255.0


opt = Adam(learning_rate = 0.00006, amsgrad=True)

# %% Build the CNN

cnn = Sequential()

cnn.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l1_l2(0, 0.03)))
cnn.add(Dropout(0.1))
#cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(16, (3, 3), activation='relu',  kernel_regularizer=l1_l2(0, 0.03)))
cnn.add(Dropout(0.1))
#cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))



cnn.add(Flatten())
cnn.add(Dense(8, activation='relu'))
cnn.add(Dropout(0.03))
cnn.add(Dense(8, activation='relu'))
cnn.add(Dropout(0.03))
cnn.add(Dense(num_classes, activation='softmax'))

'''
cnn.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model_hist = cnn.fit(
  train_images,
  y_train,
  validation_data=(valid_images, y_valid),
  epochs=150,
  batch_size=128)
'''

# %% Build the FNN


fnn = Sequential()
fnn.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l1_l2(0.003, 0.006)))
fnn.add(Dropout(0.05))
fnn.add(Dense(8, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l1_l2(0.003, 0.006)))
fnn.add(Dropout(0.05))
fnn.add(Dense(num_classes, activation='softmax'))


'''
fnn.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model_hist = fnn.fit(
  X_train,
  y_train,
  validation_data=(X_valid, y_valid),
  epochs=5250,
  batch_size=64)

'''

# %% Concatenate the Neural Networks

combinedInput = concatenate([fnn.output, cnn.output])

x = Dense(8, activation="relu", kernel_regularizer=l1_l2(0.003, 0.003))(combinedInput)
x = Dense(num_classes, activation="softmax")(x)



# The final model accepts numerical data on the MLP input and images on the CNN input, outputting a single value
model = Model(inputs=[fnn.input, cnn.input], outputs=x)

# %% Save model weights using the checkpoints

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)


# %% Compile the model


opt = Adam(learning_rate = 0.0001, amsgrad=True)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# %% Catch 'em all
model_hist = model.fit(
  [X_train, train_images],
  y_train,
  validation_data=([X_valid, valid_images], y_valid),
  epochs=550,
  batch_size=16,
  callbacks=[cp_callback])

# %% Evaluate the performances
score = model.evaluate([X_valid, valid_images], y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



