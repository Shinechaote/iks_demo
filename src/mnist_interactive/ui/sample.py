from ..ui.numberCreatorWindow import NumberCreatorWindow as ncw
import tensorflow as tf
import tkinter as tk

root = tk.Tk()
model = tf.keras.models.load_model("sampleModel.keras")
ncw = ncw(root, model)
root.mainloop()
