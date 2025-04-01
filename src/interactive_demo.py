import mnist_interactive.ui.numberCreatorWindow as mi
import tensorflow as tf
import tkinter as tk
from tkinter import ttk


# Load your pre-trained MNIST model
model = tf.keras.models.load_model('our_model.keras')

# create the tkinter window
root = tk.Tk()

# Initialize the interactive grid
grid = mi.NumberCreatorWindow(root, model=model, blur=0.3)

root.mainloop()
