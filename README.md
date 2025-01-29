# MNIST Interactive Model Analyzer

An interactive tool for experimenting with MNIST handwriting recognition models using a custom-built 28x28 drawing grid. This application allows users to draw digits and get real-time predictions from a trained MNIST model.
Of note; currently all models will be fed numpy an array of shape (1, 784), with normalised greyscale values between [0, 1] inclusive, so will only work with basic models, more compatibility will be available soon.

## Features

- Interactive 28x28 drawing grid that matches MNIST input dimensions
- Real-time prediction updates as you draw
- Blur effect to simulate natural handwriting
- Clear canvas functionality
- Grid overlay for precise pixel manipulation
- Support for custom MNIST-compatible models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LucaLow/MNIST-Interactive-Model-Analyzer.git
cd MNIST-Interactive-Model-Analyzer
```
Or install through pip:
```bash
pip install -i https://test.pypi.org/simple/ mnist-interactive
```

2. Install the required dependencies:
```bash
pip install numpy
pip install tensorflow
```

## Requirements

- Python >= 3.8
- TensorFlow >= 2.0
- NumPy
- Tkinter (usually comes with Python)

## Usage

1. Basic usage with a pre-trained model:

```python
import mnist_interactive.numberCreatorWindow as mi
import tensorflow as tf
import tkinter as tk
from tkinter import ttk


# Load your pre-trained MNIST model
model = tf.keras.models.load_model('sampleModel.keras')

# create the tkinter window
root = tk.Tk()


# Initialize the interactive grid
grid = mi.NumberCreatorWindow(root, model=model, blur=0.3)

root.mainloop()
```

2. Customize the blur effect:

```python
app = NumberCreatorWindow(root, model=model, blur=0.15)  # Adjust blur intensity
```

## Controls

- Left Click: Draw
- Right Click: Erase
- Clear Button: Reset the canvas
- Real-time predictions displayed above the canvas

## Project Structure

- `numberCreatorWindow.py`: Main application window and drawing logic
- `utils.py`: Utility functions for predictions and grid operations
- `example.py`: Example implementation
- `requirements.txt`: Required Python packages
- `setup.py`: Package installation configuration

## API Reference

### NumberCreatorWindow

```python
NumberCreatorWindow(root, model, blur=0.1)
```

Parameters:
- `root`: Tkinter root window
- `model`: TensorFlow model for digit prediction
- `blur`: Blur intensity for drawing (default: 0.1)

### Methods

- `clear_canvas()`: Clear the drawing grid
- `predict()`: Get prediction for current drawing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Luca Lowndes (Luca@Lowndes.net)

## Acknowledgments

- Built using TensorFlow and MNIST dataset
- Inspired by the need for interactive MNIST model testing tools based upon the models you create
