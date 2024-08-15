import matplotlib.pyplot as plt
import cv2 as cv
import time
from IPython.display import Image
import numpy as np

def plot_img(n, figsize,titles,imgs, n_row=1):
    """
    Plots multiple images in a single row with specified titles.

    Parameters:
    - n (int): Number of images to plot.
    - figsize (tuple): Size of the figure (width, height).
    - titles (list of str): List of titles for the images.
    - imgs (list of numpy arrays): List of images to be plotted. Images should be in BGR format.
    """
    x, y = figsize
    fig, axes = plt.subplots(n_row, n // n_row, figsize=(x, y))
    axes = axes.ravel()
    for i in range(n):
        axes[i].imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB))
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def take_pic(delay, img_path, show):
    """
    Takes a picture with video capture and saves it into a path, returns the frame

    Parameters:
    - delay (int): Waiting time in seconds to take the picture
    - img_path (str): Path of the new image
    - show (bool): Whether or not to print the image

    Returns:
    - numpy.ndarray: Image taken.
    """
    cap = cv.VideoCapture(0)
    print(f"Taking picture in {delay} seconds...")
    time.sleep(delay)
    ret, frame = cap.read()
    if ret:    
        cv.imwrite(img_path, frame)
        print(f"Image saved at {img_path}")
    else:
        print("Error: Could not capture an image.")
    cap.release()
    if show:
        Image(filename=img_path, width=250, height=250)
    return frame

def adjust_gamma(image, gamma=1.0):
    """
    Given an image and a gamma factor, returns the image with the gamma correction

    Paramters:
    - image (numpy.ndarray): Numpy ndarray of the image
    - gamma (float): gamma value

    Returns:
    - numpy.ndarray: The gamma-corrected image.
    """
    # Function gathered from pyimagesearch.com
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv.LUT(image, table) # Apply gamma correction using the lookup table

def max_pooling(image, pool_size):
    """
    Given a pooling size and an image returns the downsampled image.

    Parameters:
    - image (numpy.ndarray): Numpy ndarray of the image
    - pool_size (int): The size of the pooling window.

    Returns:
    - numpy.ndarray: The downsampled image after applying max pooling.
    """
    new_shape = (image.shape[1] // pool_size, image.shape[0] // pool_size) # Dimensions of the new image
    image_resized = cv.resize(image, (new_shape[0] * pool_size, new_shape[1] * pool_size))
    pooled_image = np.max(image_resized.reshape(new_shape[1], pool_size, new_shape[0], pool_size, -1), axis=(1, 3))

    return pooled_image