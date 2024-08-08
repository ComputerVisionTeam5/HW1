import matplotlib.pyplot as plt
import cv2 as cv

def plot_img(n, figsize,titles,imgs):
    """
    Plots multiple images in a single row with specified titles.

    Parameters:
    - n (int): Number of images to plot.
    - figsize (tuple): Size of the figure (width, height).
    - titles (list of str): List of titles for the images.
    - imgs (list of numpy arrays): List of images to be plotted. Images should be in BGR format.
    """
    x, y = figsize
    fig, axes = plt.subplots(1, n, figsize=(x, y))
    for i in range(n):
        axes[i].imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB))
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()