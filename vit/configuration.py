from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import timm

import gc
import os
import pickle as pkl
import random 

from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn import metrics

VIT_PRETRAINED_MODEL = "ViT pretrained version"
# dataset path
TRAIN_PATH = "training path"
VALID_PATH = "validation path"
TEST_PATH = "testing path"

# model and history save path
BEST_MODEL_PATH = "best model path"
HISTORY_PATH = "model history path"

# model specific global variables
IMG_SIZE = 224
BATCH_SIZE = 16
LR = 1e-05
GAMMA = 0.7
N_EPOCHS = 30

LABELS = {
    
    # "LABEL": index, Example: "NORMAL": 0,    "PNEUMONIA": 1

}

Id2LABELS = {value: key for key, value in LABELS.items()}

NUM_WORKS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE
