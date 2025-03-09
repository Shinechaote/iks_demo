import tkinter as tk
from ..ui.canvas import Canvas
from ..backends.tensorflow_backend import TensorFlowBackend
from ..backends.pytorch_backend import PyTorchBackend
import tensorflow as tf
import torch as torch
import numpy as np
from typing import Type
import os


class NumberCreatorWindow:

    # root: Tkinter root window, tk.Tk
    # model: Trained model, tf.keras.Model or torch.nn.Module
    # blur: Amount of blur to add to the drawing, float

    def __init__(self, root, model, blur=0.1, ):
        if model is None:
            raise ValueError("Model is not loaded")
        self.root = root
        self.model = model

        self.MLBackend = None
        if isinstance(model, tf.keras.Model):
            self.MLBackend = TensorFlowBackend(model)
        elif isinstance(model, torch.nn.Module):
            self.MLBackend = PyTorchBackend(model)
        else:
            raise ValueError("Model not supported")
        
        self.can = Canvas(self.root, self.MLBackend, blur)
        self.root.mainloop()




