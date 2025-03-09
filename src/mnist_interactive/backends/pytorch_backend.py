from .base_backend import BaseBackend
import numpy as np
import torch as torch


class PyTorchBackend(BaseBackend):
    def __init__(self, model) -> None:
        pass


    def predict(self, data, conversion_function=None, output_function=None):
        pass


    def gen_activations_model(self):
        pass


    def gen_activations(self, input_data):
        pass

    def predictNum(self, data, conversion_function=None, output_function=None):
        pass

    def displayableLayers(self):
        pass

