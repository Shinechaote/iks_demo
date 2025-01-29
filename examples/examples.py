# pip install -i https://test.pypi.org/simple/ mnist-interactive==1.0.4

import mnist_interactive.numberCreatorWindow as mi
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk


# Load your pre-trained MNIST model
model = tf.keras.models.load_model('sampleModel.keras')

# create the tkinter window
root = tk.Tk()


# Initialize the interactive grid
grid = mi.NumberCreatorWindow(root, blur=0.15, model=model)

root.mainloop()