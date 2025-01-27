import numpy as np
import tensorflow as tf

def predict(model, data):
    if model is None:
        print("Model is not loaded")
        return 0, 0
    x = np.array(data).reshape(1, 784)

    prediction = model.predict(x)
    return np.argmax(prediction), prediction[0][np.argmax(prediction)]