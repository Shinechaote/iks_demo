from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class BaseBackend(ABC):
    @abstractmethod
    def predict(self, data, conversion_function=None, output_function=None) -> npt.NDArray:
        pass

    @abstractmethod
    def predictNum(self, data, conversion_function=None, output_function=None):
        pass
    
    @abstractmethod # creates a model that can be used to get the activations of the layers
    def gen_activations_model(self, input_data):
        pass

    @abstractmethod # this method takes in the display_layer_outputs model from the base class
    def gen_activations(self, input_data) -> npt.NDArray:
        pass

    @abstractmethod
    def displayableLayers(self):
        pass