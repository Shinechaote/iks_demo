# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class NumberCreatorWindow:
    
    def createWindow(self, rows=28, cols=28):
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        ax.set_axis_off()

        buttonHeight = 0.8/rows
        buttonWidth = 0.8/cols

        buttons = np.empty((rows, cols), dtype=object)

        def buttonClick(event):
            pos = event.inaxes.get_label()
            i, j = map(int, pos.split(','))
            print(f'Button {i}, {j} clicked')
        
        for i in range(rows):
            for j in range(cols):
                left = 0.1 + j * buttonWidth
                bottom = 0.1 + (rows - 1 - i) * buttonHeight
                
                # Create axes for button
                button_ax = plt.axes([left, bottom, buttonWidth, buttonHeight])
                button_ax.set_label(f"{i},{j}")  # Store position in label
                
                # Create button
                btn = Button(button_ax, '')
                btn.on_clicked(buttonClick)
                buttons[i, j] = btn
        
        plt.show()
        return buttons

window = NumberCreatorWindow()
window.createWindow()



