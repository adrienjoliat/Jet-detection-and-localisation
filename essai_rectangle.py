import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.image import imread
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
from IPython.display import HTML
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Operating System Interaction
import os
import sys

# Machine Learning Frameworks
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

# Data Transformation and Augmentation (not all of these transformations were finally used)
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, \
    RandomVerticalFlip, ColorJitter, RandomAffine, RandomPerspective, RandomResizedCrop, \
    GaussianBlur, RandomAutocontrast
from torchvision.transforms import functional as F

# Model Building and Initialization
import torch.nn as nn
from torch.nn.init import kaiming_normal_

# Data Loading and Dataset Handling
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Subset
from PIL import Image
import json

# Cross-Validation and Metrics
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, confusion_matrix
from scipy.special import expit as sigmoid

# Visualization and Display
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from IPython.display import HTML
from astropy.visualization import ImageNormalize, SqrtStretch
import seaborn as sns
import sunpy.visualization.colormaps as cm

# Miscellaneous
import random
from tqdm import tqdm



class RectangleDrawer:
    def __init__(self, image):
        self.image = image
        self.fig, self.ax = plt.subplots()
        vmin, vmax = np.percentile(self.image, [1, 99.9])
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        self.ax.imshow(np.zeros(self.image.shape[:2]), cmap='sdoaia304', norm=norm)
        self.rect = None
        self.ax.set_aspect('auto')  # Set aspect ratio to match the image
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.dblclick:
            return

        if event.button == 1:
            if self.rect:
                self.rect.remove()
            self.start_point = (event.xdata, event.ydata)
        elif event.button == 3:
            self.end_point = (event.xdata, event.ydata)
            self.draw_rectangle()

    def draw_rectangle(self):
        x = min(self.start_point[0], self.end_point[0])
        y = min(self.start_point[1], self.end_point[1])
        width = abs(self.start_point[0] - self.end_point[0])
        height = abs(self.start_point[1] - self.end_point[1])
        self.rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        self.ax.add_patch(self.rect)
        self.fig.canvas.draw()

    def show(self):
        plt.show()

image_path = "./data/data separated/data1/1.npz"  # Replace "your_image.jpg" with the path to your image
archive = np.load("./data/data separated/data1/1.npz")
array = archive["arr_0"]
image=array[:,:,0]
print(image.shape)
drawer = RectangleDrawer(image)
drawer.show()
print(drawer.rect)
