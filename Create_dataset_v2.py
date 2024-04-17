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
    def __init__(self, images):
        self.images = images
        self.fig, self.ax = plt.subplots()
        self.vmin, self.vmax = np.percentile(images, [1, 99.9])
        self.norm = ImageNormalize(vmin=self.vmin, vmax=self.vmax, stretch=SqrtStretch())
        self.current_index = 0
        self.image_plot = self.ax.imshow(images[:, :, self.current_index], cmap='sdoaia304', norm=self.norm)
        self.rect = None
        self.ax.set_aspect('auto')  # Set aspect ratio to match the image
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.boxes_per_image = []  # List to store boxes for each image

    def onclick(self, event):
        if event.dblclick:
            return

        if event.button == 1:
            self.start_point = (event.xdata, event.ydata)
            
        elif event.button == 3:
            self.end_point = (event.xdata, event.ydata)
            self.draw_rectangle()

    def draw_rectangle(self):
        x = min(self.start_point[0], self.end_point[0])
        y = min(self.start_point[1], self.end_point[1])
        width = abs(self.start_point[0] - self.end_point[0])
        height = abs(self.start_point[1] - self.end_point[1])
        self.rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='w', facecolor='none')
        self.ax.add_patch(self.rect)
        self.fig.canvas.draw()
        self.boxes_per_image.append(torchvision.ops.box_convert(torch.tensor([x, y, width, height]), "xywh", "xyxy").tolist())

    def draw_rectangles_for_all_images(self):
        for i in range(self.images.shape[2]):
            self.ax.imshow(self.images[:, :, i], cmap='sdoaia304', norm=self.norm)
            plt.title(f'Image {i+1}/{self.images.shape[2]}')
            self.fig.canvas.draw()
            self.boxes_per_image = []  # Reset boxes for each image
            plt.show()

    def save_boxes_per_image(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        #for i, boxes in enumerate(self.boxes_per_image):
            #np.savez(os.path.join(folder_path, f"image_{i+1}_boxes"), boxes=np.array(boxes))


# Load images 
number = 161
seq = f"./data/data separated/data_jet_image/{number}.npz"
images = np.load(seq)["arr_0"]

# Create RectangleDrawer instance with the sequence of images
drawer = RectangleDrawer(images)
drawer.draw_rectangles_for_all_images()
drawer.save_boxes_per_image("./data_boxes/events")

