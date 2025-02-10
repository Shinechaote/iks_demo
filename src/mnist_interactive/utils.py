import numpy as np
import tensorflow as tf
import tkinter as tk
import tkinter.ttk as ttk
import math


def predict(model, data, conversion_function, output_function):
    if model is None:
        print("Model is not loaded")
        return 0, 0
    
    x = None

    # If no conversion function is provided, assume the data is to be reshaped to (1, 784)
    if conversion_function is None:
        x = np.array(data).reshape(1, 784)
    else:
        x = conversion_function(data)

    prediction = model.predict(x)

    # If no output function is provided, revert to default where the output is the index of the highest value
    if conversion_function is None:
        return np.argmax(prediction), prediction[0][np.argmax(prediction)]
    
    return output_function(prediction)

def drawGridLines(canvas):
    for i in range(29):
        x = i * 10
        canvas.create_line(x, 0, x, 280, fill="gray")
        canvas.create_line(0, x, 280, x, fill="gray")

def addBlur(grid_data, canvas, grid_y, grid_x, x_1, x_2, y_1, y_2, blur):
    if grid_y >= 28 or grid_x >= 28 or grid_y < 0 or grid_x < 0 or grid_data[grid_y, grid_x] >= 1:
        return grid_data
    grid_data[grid_y, grid_x] += 0.1 * blur
    if grid_data[grid_y, grid_x] > 1:
        grid_data[grid_y, grid_x] = 1
    canvas.create_rectangle(
        x_1, y_1, x_2, y_2,
        fill="#" + f"{int(grid_data[grid_y, grid_x] * 255):02X}" * 3,
        outline="grey"
    )
    return grid_data

def allLayerOutputs(model, data): # in the future this function will include a conversion function and will be compatible with convulutional layers
    if model is None:
        print("Model is not loaded")
        return 0, 0
    
    x = None

    x = np.array(data).reshape(1, 784)

    prediction = model.predict(x)

    return np.argmax(prediction), prediction[0][np.argmax(prediction)]
    
    return output_function(prediction)

def display_model_internals(model, root):
    FRAME_WIDTH = 1000
    FRAME_HEIGHT = 500
    
    layer_spacing = 40
    neuron_spacing = 3
    radius = 3

    MAX_NEURONS = math.floor(FRAME_WIDTH / neuron_spacing)-4

    print("Displaying model internals...")

    # Create a frame to hold the canvas
    frame = tk.Frame(root, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    frame.grid(row=0, column=1, sticky="nsew")
    frame.pack_propagate(False)  # Prevent resizing

    # Create a canvas for drawing neurons
    canvas = tk.Canvas(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT, bg="white")
    canvas.pack(fill="both", expand=True)


    for layer_number, layer in enumerate([layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]):
            y = layer_spacing * layer_number + 50
            for neuron_number in range(MAX_NEURONS if layer.units > MAX_NEURONS else layer.units):
                x = neuron_spacing * neuron_number - ((MAX_NEURONS if layer.units > MAX_NEURONS else layer.units) * neuron_spacing / 2) + (FRAME_WIDTH / 2)
                print(f"Drawing neuron at ({x}, {y})")
                canvas.create_oval(
                    x-radius, 
                    y-radius, 
                    x+radius, 
                    y+radius, 
                    fill="blue"
                )
            

    canvas.update_idletasks()  # Ensure the canvas updates
