import numpy as np
import tensorflow as tf

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