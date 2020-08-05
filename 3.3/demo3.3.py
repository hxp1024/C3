import tensorflow as tf
import numpy as np


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

train_x, test_x = train_x / 255, test_x / 255

dataset_x = tf.data.Dataset.from_tensor_slices(train_x)
dataset_y = tf.data.Dataset.from_tensor_slices(train_y)

dataset = tf.data.Dataset.zip((dataset_x, dataset_y)).shuffle(60000).repeat().batch(batch_size=64)

for ele in dataset:
    print(ele)


