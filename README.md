# Overview

The MNIST Interactive Library is a Python-based tool designed for users working with the MNIST dataset and TensorFlow models. This library provides an interactive 28x28 grid where users can simulate drawing digits, submit their input to a pre-trained model, and instantly view the model's predictions along with its confidence scores.

Key Features

1. Interactive 28x28 Grid

A fully interactive GUI built using Tkinter.

Users can toggle individual grid cells to simulate drawing digits.

Includes buttons to "Clear" the grid and "Submit" the drawn input to the model.

2. Model Integration

Accepts any pre-trained TensorFlow/Keras model as input.

Converts the grid state into a 28x28 grayscale NumPy array suitable for MNIST model predictions.

3. Real-Time Predictions

Displays the predicted digit based on the drawn input.

Provides the confidence level of the model's prediction.

4. User-Friendly Design

Simple and intuitive interface suitable for beginners and experts alike.

Designed to facilitate experimentation and learning with neural networks.

Installation

Clone the repository:

git clone https://github.com/your-username/mnist-interactive.git
cd mnist-interactive

Install the required dependencies:

pip install -r requirements.txt

Usage

Import the library and load your pre-trained model:

from tensorflow.keras.models import load_model
from mnist_interactive.interactive_grid import InteractiveGrid

# Load your pre-trained MNIST model
model = load_model("path_to_your_model.h5")

# Initialize the interactive grid
grid = InteractiveGrid(model)
grid.run()

Interact with the grid:

Toggle cells to simulate drawing a digit.

Click "Submit" to see the model's prediction and confidence.

Use the "Clear" button to reset the grid.

Example

from tensorflow.keras.models import load_model
from mnist_interactive.interactive_grid import InteractiveGrid

# Load a model
model = load_model("mnist_model.h5")

# Start the grid interface
grid = InteractiveGrid(model)
grid.run()

Requirements

Python 3.8+

TensorFlow 2.0+

NumPy

Tkinter (included with most Python installations)

Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests. If you encounter any issues or have suggestions, please open an issue on GitHub.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments

MNIST Dataset: LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998.

TensorFlow: An open-source machine learning framework by Google.
