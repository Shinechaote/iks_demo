import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from DatasetLoader import MNISTDataLoader



trainingLoader = MNISTDataLoader("SampleDataset/Train/train-images-idx3-ubyte", "SampleDataset/Train/train-labels-idx1-ubyte")
trainingLoader.readData()
testLoader = MNISTDataLoader("SampleDataset/Test/t10k-images-idx3-ubyte", "SampleDataset/Test/t10k-labels-idx1-ubyte")
testLoader.readData()


model = Sequential ([
    Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
], name="test")

optimizer = Adam(learning_rate=0.01, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Split the training data into training and validation data
train_data, val_data, train_labels, val_labels = train_test_split(trainingLoader.data, trainingLoader.labels, test_size=0.1, random_state=42)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(train_data, train_labels, epochs=4, batch_size=32, validation_data=(val_data, val_labels), callbacks=[early_stopping])

# Evaluate the model
print(f"Loss, Accuracy = {model.evaluate(testLoader.data, testLoader.labels)}")
model.save("sampleModel.keras")