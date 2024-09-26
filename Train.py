import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

from InkDataset import MathWritingDataset
from Tokenizer import LaTeXTokenizer

# Setup vocab, collate_fn, transform -> datasets
data_path = 'HME_Training/data/mathwriting_2024'
cache_path = 'HME_Training/data/full_cache'

tokenizer = LaTeXTokenizer()
latexList = []
data_dir = Path(data_path)
for latex_file in data_dir.glob("train/*.txt"):
    with open(latex_file) as f:
            latexList.append(f.read())
tokenizer.build_vocab(latexList) # Takes list as input to assign IDs

class collate_fn():
     pass

transform = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
    transforms.ToTensor()
])

train_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='train', transform=transform)
valid_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='valid', transform=transform)
test_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='test', transform=transform)

def main():
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=5) 
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    
    



