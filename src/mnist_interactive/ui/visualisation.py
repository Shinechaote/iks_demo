import numpy as np
import math
import tkinter as tk
import tkinter.ttk as ttk

class visualisation():
    def __init__(self, MLBackend, frame_width = 750, frame_height = 300):
        self.MLBackend = MLBackend
        self.FRAME_WIDTH = frame_width
        self.FRAME_HEIGHT = frame_height


    def display_model_internals(self, root, data):
        layer_spacing = self.FRAME_HEIGHT / len(self.MLBackend.displayableLayers())
        neuron_spacing = 10
        radius = 4

        MAX_NEURONS = math.floor(self.FRAME_WIDTH / neuron_spacing)-4

        # Create a frame to hold the canvas
        frame = tk.Frame(root, width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.pack_propagate(False)  # Prevent resizing

        # Create a canvas for drawing neurons
        canvas = tk.Canvas(frame, width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT, bg="grey")
        canvas.pack(fill="both", expand=True)

        # Get the activations of the model
        activations = self.MLBackend.gen_activations(data)
        #Draw the neurons
        for layer_number, layer in enumerate(self.MLBackend.displayableLayers()):
                    y = layer_spacing * layer_number + 50
                    act_max = np.max(activations[layer_number])
                    act_min = np.min(activations[layer_number])
                    for neuron_number in range(MAX_NEURONS if layer.units > MAX_NEURONS else layer.units):
                        act_max = max(act_max-act_min, 1e-10)
                        x = neuron_spacing * neuron_number - ((MAX_NEURONS if layer.units > MAX_NEURONS else layer.units) * neuron_spacing / 2) + (self.FRAME_WIDTH / 2)
                        colour_num = (activations[layer_number][0][neuron_number] - act_min) / (act_max - act_min) * 255
                        if colour_num < 0:
                            colour_num = 0
                        if colour_num > 255 or math.isnan(colour_num):
                            colour_num = 255
                        
                        canvas.create_oval(
                            x-radius, 
                            y-radius, 
                            x+radius, 
                            y+radius, 
                            fill="#"+f"{int(colour_num):02X}"*3
                        )

    def addBlur(self, grid_data, canvas, grid_y, grid_x, bx1, bx2, by1, by2, blur):
        if grid_y >= 28 or grid_x >= 28 or grid_y < 0 or grid_x < 0 or grid_data[grid_y, grid_x] >= 1:
            return grid_data
        grid_data[grid_y, grid_x] += 0.1 * blur
        if grid_data[grid_y, grid_x] > 1:
            grid_data[grid_y, grid_x] = 1
        canvas.create_rectangle(
            bx1, by1, bx2, by2,
            fill="#" + f"{int(grid_data[grid_y, grid_x] * 255):02X}" * 3,
            outline="grey"
        )
        return grid_data
    