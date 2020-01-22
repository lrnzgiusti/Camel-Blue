# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:41:23 2020

@author: logiusti
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:02:20 2020

@author: logiusti
"""
import tensorflow as tf
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.models import Model

batch_size = 128
num_classes = 10
epochs = 12



x_train_data = pd.read_csv(r'mnist_train.csv').to_numpy().astype('float32')
x_test_data = pd.read_csv(r'mnist_test.csv').to_numpy().astype('float32')

num_raw_data_features = x_train_data.shape[1]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train_data, x_train ,y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_train_data, x_train ,y_train)).batch(32)

cnn = tf.keras.models.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    #Dense(num_classes, activation='softmax')
    ])


# The final model accepts numerical data on the MLP input and images on the CNN input, outputting a single value

fnn = tf.keras.models.Sequential([
  Dense(128, input_shape=(num_raw_data_features,),activation='relu'),
  Dropout(0.2),
  #tensorflow.keras.layers.Dense(num_classes, activation='softmax')
])


combinedInput = concatenate([fnn.output, cnn.output])

x = Dense(4, activation="relu", name='dense3')(combinedInput)
x = Dense(10, activation="softmax", name='dense4')(x)



# The final model accepts numerical data on the MLP input and images on the CNN input, outputting a single value
model = Model(inputs=[fnn.input, cnn.input], outputs=x)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(data, images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model([data, images])
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(data, images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model([data, images])
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)






EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for data, images, labels in train_ds:
    train_step(data, images, labels)

  for test_data, test_images, test_labels in test_ds:
    test_step(test_data, test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))