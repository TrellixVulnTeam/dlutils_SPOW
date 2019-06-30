import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import itertools

def imshow(image):
    plt.imshow(image)

def plot_dotted(ax, x_data, y_data, label="dotted data"):
    return ax.plot(x_data, y_data, 'ro', label=label)

def plot_lined(ax, x_data, y_data, label="lined data"):
    return ax.plot(x_data, y_data, label=label)

def add2gif(fig, images):
    # Used to return the plot as an image rray
    fig.canvas.draw();       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)
    plt.close(fig)
