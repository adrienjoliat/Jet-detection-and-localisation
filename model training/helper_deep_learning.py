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

#Hand-made functions
from helper_deep_learning import *




def train_epoch(model, optimizer, scheduler, criterion, train_loader, device, threshold):
    model.train()
    correct, train_loss, total_length = 0, 0, 0

    for i, (data, target) in enumerate(train_loader):

        #MOVING THE TENSORS TO THE CONFIGURED DEVICE
        data, target = data.to(device), target.to(device).unsqueeze(1).float()

        #FORWARD PASS
        output = model(data)
        loss = criterion(output, target)

        #BACKWARD AND OPTIMIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # PREDICTIONS
        pred = (torch.sigmoid(output) >= threshold).float()

        # PERFORMANCE CALCULATION
        train_loss += loss.item() * len(data)
        total_length += len(data)
        correct += pred.eq(target.view_as(pred)).sum().item()

    
    scheduler.step()
    train_loss = train_loss / total_length
    train_acc = correct / total_length

    return train_loss, train_acc, scheduler.get_last_lr()[0]


def predict(model, train_loader, criterion, device, threshold):
    model.eval()
    correct, val_loss, total_length = 0, 0, 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in train_loader:

            #MOVING THE TENSORS TO THE CONFIGURED DEVICE
            data, target = data.to(device), target.to(device).unsqueeze(1).float()

            #FORWARD PASS
            output = model(data)
            loss = criterion(output, target)
            # PREDICTIONS
            pred = (torch.sigmoid(output) >= threshold).float()

            # PERFORMANCE CALCULATION
            val_loss += loss.item() * len(data)
            total_length += len(data)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())

    val_loss = val_loss / total_length
    val_acc = correct / total_length

    return val_loss, val_acc, np.array(all_preds), np.array(all_targets)


def test(model, train_loader, criterion, device, threshold):
    model.eval()
    correct, val_loss, total_length = 0, 0, 0
    all_preds = []
    all_targets = []
    all_out = []

    with torch.no_grad():
        for data, target in train_loader:

            #MOVING THE TENSORS TO THE CONFIGURED DEVICE
            data, target = data.to(device), target.to(device).unsqueeze(1).float()

            #FORWARD PASS
            output = model(data)
            loss = criterion(output, target)
            # PREDICTIONS
            pred = (torch.sigmoid(output) >= threshold).float()

            # PERFORMANCE CALCULATION
            val_loss += loss.item() * len(data)
            total_length += len(data)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())
            all_out.extend(output.view(-1).cpu().numpy())

    val_loss = val_loss / total_length
    val_acc = correct / total_length

    return val_loss, val_acc, np.array(all_preds), np.array(all_targets), np.array(all_out)

#Learning rate visualization functions:
def plot_lr(epochs, lrs):
    # Creating subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Linear scale plot
    ax1.plot(epochs, lrs, marker='o')
    ax1.set_title('Learning Rate per Epoch (Linear Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_xticks(epochs)
    ax1.grid(True)

    # Log scale plot
    ax2.plot(epochs, lrs, marker='o')
    ax2.set_title('Learning Rate per Epoch (Log Scale)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.set_xticks(epochs)
    ax2.grid(True)

    # Display the plots
    plt.show()


def get_lrs(num_epochs, optimizer, scheduler):
    epochs = []
    lrs = []
    for epoch in range(num_epochs):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epochs.append(epoch + 1)
        lrs.append(current_lr)
    return epochs, lrs