from .base_backend import BaseBackend
import numpy as np
import tensorflow as tf


class TensorFlowBackend(BaseBackend):
    def __init__(self, model) -> None:
        self.model = model
        self.activation_model = self.gen_activations_model()


    def predict(self, data, conversion_function=None, output_function=None):
        if self.model is None:
            print("Model is not loaded")
            return 0, 0
        
        # conversion functions can be utilized when specific models used aren't explicitely compatible with the input data
        if conversion_function is None:
            data = np.array(data).reshape(1, 784)
        else:
            data = conversion_function(data)
        prediction = self.model.predict(data)

        # If no output function is provided, revert to default where the output is the index of the highest value
        if output_function is None:
            return np.argmax(prediction), prediction[0][np.argmax(prediction)]
        
        return output_function(prediction)


    def gen_activations_model(self):
        dense_layers = [layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Dense)]
        return tf.keras.Model(inputs=self.model.inputs, outputs=[layer.output for layer in dense_layers])


    def gen_activations(self, input_data):
        input_data = input_data.reshape(1, 784)
        return self.activation_model.predict(input_data)

    def predictNum(self, data, conversion_function=None, output_function=None):
        confidence = 0
        prediction = 0
        if self.model is None:
            print("Model is not loaded")
            return prediction, confidence
        
        if conversion_function is None:
            data = np.array(data).reshape(1, 784)
        else:
            data = conversion_function(data)
        prediction = self.model.predict(data)

        if output_function is None:
            return np.argmax(prediction), prediction[0][np.argmax(prediction)]
        
        return output_function(prediction)

    def displayableLayers(self):
        return [layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Dense)]

