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
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

from InkDataset import MathWritingDataset
from Tokenizer import LaTeXTokenizer

from Models import Model_1
from torch.nn import Softmax

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup vocab, collate_fn, transform -> datasets

data_path = Path(r'HME_Training\data\mathwriting_2024_excerpt')
cache_path = Path(r'HME_Training\data\excerpt_cache')

tokenizer = LaTeXTokenizer()
latexList = []
for latex_file in cache_path.glob("train/*.txt"):
    with open(latex_file) as f:
            latexList.append(str(f.read()))
tokenizer.build_vocab(latexList) # Takes list as input to assign IDs

def collate_fn(batch):
    images, labels = zip(*batch)

    tgt = pad_sequence(labels, batch_first=True).long() # Pads all sequences to the max of the batch
    seq_len = tgt.size(1) # Max sequence length - because tgt is of size (batch x max_length)
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1) # ATTENTION MASK, see https://pytorch.org/docs/stable/generated/torch.triu.html
   
    images = torch.stack(images)  # tensor of shape (batch_size, C, H, W)

    return images, tgt, tgt_mask

transform = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
    transforms.ToTensor()
])

train_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='train', transform=transform)
valid_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='valid', transform=transform)
test_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='test', transform=transform)

def main(num_epochs, model_in, LR, dropout):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn) 
    val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    accuracy = Accuracy(task='multiclass', num_classes=len(tokenizer.vocab))
    accuracy = accuracy.to(device)
    model = model_in
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
   
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        accuracy.reset()
        
        for images, tgt, tgt_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}: training loop"):
            images = images.to(device)
            tgt = tgt.to(device)
            tgt_mask = tgt_mask.to(device)

            outputs, tgts_for_loss = model(images, tgt, tgt_mask) # forward
            print(outputs[0])
            break


if __name__ == '__main__': 
    main(
        num_epochs = 1,
        model_in = Model_1(vocab_size=len(tokenizer.vocab), d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3),
        LR = 1e-4,
        dropout=0
    )
    
    



