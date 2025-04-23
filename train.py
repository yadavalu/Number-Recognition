import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Neural network to recognise handwritten digits
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Reshape
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

batch_size = 10
n_batches = len(train_images) // batch_size

# Initialize model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(train_images.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])

# model_2 = model

# Compile model
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# model_2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])


# Callbacks
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='/model_data', update_freq=int(n_batches/10))

# Train model
model.fit(
    train_images, 
    train_labels,  
    epochs=10,
    validation_data=(test_images, test_labels),
)

model.save('model/my_model.keras')
model.summary()
