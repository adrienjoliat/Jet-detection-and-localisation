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
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score
from scipy.special import expit as sigmoid

# Visualization and Display
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from IPython.display import HTML

# Miscellaneous
import random
from tqdm import tqdm



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
    f1 = []

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

def get_param(result_dict, i):
    """
    Gets the parameters name as strings to use as label for the plots

    """

    hyperparams_name = f"hyperparams_{i}"
    hyperparams = result_dict[hyperparams_name]

    lr = hyperparams[0]
    scheduler_type = hyperparams[1]

    if scheduler_type == "<class 'torch.optim.lr_scheduler.ExponentialLR'>":
      scheduler_type = 'Exponential LR'

    elif scheduler_type == "<class 'torch.optim.lr_scheduler.CosineAnnealingLR'>":
      scheduler_type = 'Cosine annealing LR'

    elif scheduler_type == "<class 'torch.optim.lr_scheduler.ReduceLROnPlateau'>":
      scheduler_type = 'Reduce LR on plateau'

    optimizer = hyperparams[2]

    if optimizer == "<class 'torch.optim.sgd.SGD'>":
      optimizer = 'SGD'

    elif optimizer ==  "<class 'torch.optim.adamw.AdamW'>":
      optimizer = 'Adam'

    return lr, scheduler_type, optimizer