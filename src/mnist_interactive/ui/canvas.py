import numpy as np
import tkinter as tk
from tkinter import ttk
from typing import Type
from ..backends.base_backend import BaseBackend
from .visualisation import visualisation

class Canvas:
    def __init__(self, root, MLBackend: Type[BaseBackend]):
        self.root = root
        self.visualisation = visualisation(MLBackend)
        self.MLBackend = MLBackend
        self.root.title("MNIST Drawing App")

        self.label = ttk.Label(self.root, text="Draw a number")
        
        self.drawing = False
        self.removing = False
        self.blur = 0.1
        
        # the following variables may eventually be moved up stream/into a different file and class.
        self.drawing = False
        self.removing = False
        self.grid_data = np.zeros((28, 28), dtype=np.float32)
        
        self._create_widgets()
        self._create_canvas()

        self.predict()


        

    def drawGridLines(self, canvas):
        for i in range(29):
            x = i * 10
            canvas.create_line(x, 0, x, 280, fill="gray")
            canvas.create_line(0, x, 280, x, fill="gray")


    def _create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.clear_button = ttk.Button(self.main_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.S), pady=10)

        self.prediction_label = ttk.Label(self.main_frame, text="Prediction: ")
        self.prediction_label.grid(row=1, column=2, padx=5, pady=10, sticky=(tk.S))

    def _create_canvas(self):
        self.canvas = tk.Canvas(self.main_frame, width=280, height=280, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.N, tk.S, tk.E, tk.W))

        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        self.canvas.bind('<Button-2>', self.start_removing)
        self.canvas.bind('<B2-Motion>', self.remove)
        self.canvas.bind('<ButtonRelease-1>', self.stop_removing)

        self.drawGridLines(self.canvas)

    def start_drawing(self, event):
        self.drawing = True
        self.draw(event)

    def stop_drawing(self, event):
        self.drawing = False
        self.predict()

    def draw(self, event):
        if not self.drawing:
            return
    
        grid_x = event.x // 10
        grid_y = event.y // 10
        
        if 0 <= grid_x < 28 and 0 <= grid_y < 28:
            x1 = grid_x * 10
            y1 = grid_y * 10
            x2 = x1 + 10
            y2 = y1 + 10
            if self.grid_data[grid_y, grid_x] != 1.0:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#FFFFFF", outline="grey")
                self.grid_data[grid_y, grid_x] = 1.0

            blur_coords = [
                (grid_y-1, grid_x, x1, x2, y1-10, y2-10, self.blur),  # Top
                (grid_y+1, grid_x, x1, x2, y1+10, y2+10, self.blur),  # Bottom
                (grid_y, grid_x-1, x1-10, x2-10, y1, y2, self.blur),  # Left
                (grid_y, grid_x+1, x1+10, x2+10, y1, y2, self.blur),  # Right
                (grid_y-1, grid_x-1, x1-10, x2-10, y1-10, y2-10, self.blur*0.75),  # Top-Left
                (grid_y-1, grid_x+1, x1+10, x2+10, y1-10, y2-10, self.blur*0.75),  # Top-Right
                (grid_y+1, grid_x-1, x1-10, x2-10, y1+10, y2+10, self.blur*0.75),  # Bottom-Left
                (grid_y+1, grid_x+1, x1+10, x2+10, y1+10, y2+10, self.blur*0.75)   # Bottom-Right
            ]

            for y, x, bx1, bx2, by1, by2, blur in blur_coords:
                # add blur
                self.grid_data = self.visualisation.addBlur(
                    self.grid_data, self.canvas,
                    y, x, bx1, bx2, by1, by2,
                    blur
                )  

    def start_removing(self, event):
        self.removing = True
        self.remove(event)
    
    def stop_removing(self, event):
        self.removing = False
        self.predict()

    def remove(self, event):
        if not self.removing:
            return
    
        grid_x = event.x // 10
        grid_y = event.y // 10

        if 0 <= grid_x < 28 and 0 <= grid_y < 28:
            x1 = grid_x * 10
            y1 = grid_y * 10
            x2 = x1 + 10
            y2 = y1 + 10
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#000000", outline="grey")
            self.grid_data[grid_y, grid_x] = 0
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid_data = np.zeros((28, 28), dtype=np.float32)
        self.drawGridLines(self.canvas)
        self.predict()

    def predict(self):
        prediction, confidence = self.MLBackend.predict(data=self.grid_data)
        self.prediction_label.config(text=f"Prediction: {prediction}\nConfidence: {confidence:.2f}")
        self.visualisation.display_model_internals(self.root, self.grid_data)
        