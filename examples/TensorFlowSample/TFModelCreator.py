import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras im



# model = Sequential ([
#     Input(shape=(784,)),
#     tf.keras.layers.Dense(128, activation="relu"),
#     Dropout(0.2),
#     tf.keras.layers.Dense(64, activation="relu"),
#     Dropout(0.2),
#     tf.keras.layers.Dense(10, activation="softmax")
# ], name="test")

# optimizer = Adam(learning_rate=0.01, clipnorm=1.0)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train_data, val_data, train_labels, val_labels = train_test_split(trainingLoader.data, trainingLoader.labels, test_size=0.1, random_state=42)

# # Early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # Train the model
# model.fit(train_data, train_labels, epochs=4, batch_size=32, validation_data=(val_data, val_labels), callbacks=[early_stopping])

# # Check model predictions after training
# predictions = model.predict(testLoader.data)
# print("Model predictions after training:", predictions.max(axis=1).argmin())

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(testLoader.data, testLoader.labels)
# print(model.predict(testLoader.data)[0].argmax())
# testLoader.VisualiseSample(0)