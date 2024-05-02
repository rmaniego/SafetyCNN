import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

# Constants
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
MODELS_DIR = "models"
TRAINING_DIR = "dataset/train"
MODEL_NAME = "models/cnn.vgg16.keras"

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# Data augmentation and normalization
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data
dataset = generator.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define number of classes
NUM_CLASSES = len(dataset.class_indices)

# Model architecture
model = Sequential()

model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())

# Dropout is typically applied after the fully connected layers.
# Value range from 0.2-0.5, with 0.5 as ideal to avoid overfitting in smaller datasets.
model.add(Dense(4096, activation="relu"))
model.add(Dense(4096, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output layer
model.add(Dense(NUM_CLASSES, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(dataset, epochs=EPOCHS)

# Save the model
model.save(MODEL_NAME)
