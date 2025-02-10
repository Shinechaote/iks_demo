import numberCreatorWindow as mi
import tensorflow as tf
import tkinter as tk
from tkinter import ttk

# To install the package:
# pip install -i https://test.pypi.org/simple/mnist-interactive

# Load your pre-trained MNIST model
model = tf.keras.models.load_model('sampleModel.keras')

# create the tkinter window
root = tk.Tk()


# Initialize the interactive grid
grid = mi.NumberCreatorWindow(root, blur=0.3, model=model)

root.mainloop()