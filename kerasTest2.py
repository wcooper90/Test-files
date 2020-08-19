# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

print(len(train_labels))


train_images = train_images / 255.0

test_images = test_images / 255.0


model = keras.Sequential()

conv_1 = keras.layers.Conv2D(input_shape = (None, None, 3),
                             filters = 92,
                             kernel_size=(7,7),
                             strides=(3,3),
                             padding='valid',
                             activation = 'relu',
                             kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                             # kernel_regularizer=tf.keras.regularizers.12(0.001),
                             bias_initializer='zeros',
                             bias_regularizer=None,
                             data_format='channels_last')

conv_2 = keras.layers.Conv2D(filters = 128,
                             kernel_size=(5,5),
                             strides=(3,3),
                             padding='valid',
                             activation = 'relu',
                             kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                             # kernel_regularizer=tf.keras.regularizers.12(0.001),
                             bias_initializer='zeros',
                             bias_regularizer=None,
                             data_format='channels_last')

conv_3 = keras.layers.Conv2D(filters = 160,
                             kernel_size=(3,3),
                             strides=(3,3),
                             padding='valid',
                             activation = 'relu',
                             kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                             # kernel_regularizer=tf.keras.regularizers.12(0.001),
                             bias_initializer='zeros',
                             bias_regularizer=None,
                             data_format='channels_last')

conv_4 = keras.layers.Conv2D(filters = 192,
                             kernel_size=(3,3),
                             strides=(2,2),
                             padding='valid',
                             activation = 'relu',
                             kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                             # kernel_regularizer=tf.keras.regularizers.12(0.001),
                             bias_initializer='zeros',
                             bias_regularizer=None,
                             data_format='channels_last')

conv_5 = keras.layers.Conv2D(filters = 256,
                             kernel_size=(3,3),
                             strides=(2,2),
                             padding='valid',
                             activation = 'relu',
                             kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                             # kernel_regularizer=tf.keras.regularizers.12(0.001),
                             bias_initializer='zeros',
                             bias_regularizer=None,
                             data_format='channels_last')



pool_1 = keras.layers.MaxPooling2D(pool_size=(5,5),
                                    strides=(5,5),
                                    data_format='channels_last')

pool_2 = keras.layers.MaxPooling2D(pool_size=(3,3),
                                    strides=(3,3),
                                    data_format='channels_last')

pool_3 = keras.layers.MaxPooling2D(pool_size=(3,3),
                                    strides=(3,3),
                                    data_format='channels_last')


dropout1 = tf.keras.layers.Dropout(0.5)
dropout2 = tf.keras.layers.Dropout(0.5)


fc_layer1 = keras.layers.Dense(units=512,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=tf.keras.initializers.normal(stddev=0.005),
                  bias_initializer='zeros',
                  # kernel_regularizer=tf.keras.regularizers.12(1e-4),
                  # bias_regularizer=tf.keras.regularizers.12(1e-4),
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None)

fc_layer2 = keras.layers.Dense(units=1024,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=tf.keras.initializers.normal(stddev=0.005),
                  bias_initializer='zeros',
                  # kernel_regularizer=tf.keras.regularizers.12(1e-4),
                  # bias_regularizer=tf.keras.regularizers.12(1e-4),
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None)

fc_layer3 = keras.layers.Dense(units=2,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=tf.keras.initializers.normal(stddev=0.005),
                  bias_initializer='zeros',
                  # kernel_regularizer=tf.keras.regularizers.12(1e-4),
                  # bias_regularizer=tf.keras.regularizers.12(1e-4),
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None)

model.add(conv_1)
model.add(pool_1)

print(model.output_shape)

model.add(conv_2)
model.add(pool_2)
print(model.output_shape)



model.add(conv_3)
model.add(conv_4)
print(model.output_shape)


model.add(conv_5)
model.add(pool_3)

model.add(keras.layers.Flatten())
print(model.output_shape)


model.add(fc_layer1)
model.add(dropout1)

model.add(fc_layer2)
model.add(dropout2)

model.add(fc_layer3)


# model.add(keras.layers.Flatten(input_shape=(28,28)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(56, activation='relu'))
# model.add(keras.layers.Dense(10))


# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Conv2D(filters=84, kernel_size=(3, 3), strides=(1,1), padding='valid'),
#     keras.layers.Dense(56, activation='relu'),
#     keras.layers.Dense(10)
# ])



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



model.fit(train_images, train_labels, epochs=10)



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)



probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])



predictions = probability_model.predict(test_images)


print(predictions[0])
