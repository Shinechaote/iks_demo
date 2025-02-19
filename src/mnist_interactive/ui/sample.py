import tkinter as tk
from ..ui.canvas import Canvas
from ..backends.tensorflow_backend import TensorFlowBackend
import tensorflow as tf

root = tk.Tk()
model = tf.keras.models.load_model("sampleModel.keras")
MLBackend = TensorFlowBackend(model)
can = Canvas(root, MLBackend)
root.mainloop()
