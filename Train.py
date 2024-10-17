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
from torchvision.utils import make_grid
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

from InkDataset import MathWritingDataset
from Tokenizer import LaTeXTokenizer

from Models import Model_1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup vocab, collate_fn, transform -> datasets

data_path = Path(r'data\mathwriting_2024')
cache_path = Path(r'data\full_cache')

tokenizer = LaTeXTokenizer()
latexList = []
for latex_file in cache_path.glob("valid/*.txt"):
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

def main(num_epochs, model_in, LR, dropout, experimentNum):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=3 , collate_fn=collate_fn) 
    val_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    accuracy = Accuracy(task='multiclass', num_classes=len(tokenizer.vocab))
    accuracy = accuracy.to(device)
    model = model_in
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index

    train_accs, train_losses, val_accs, val_losses = [], [], [], []
    
    '''
    for images, tgt, tgt_mask in train_loader:
        plt.imshow(make_grid(images, nrow=4).permute(1, 2, 0))
        plt.axis("off")
        plt.show()
        for t in tgt:
            print(tokenizer.decode(t.tolist()))
        break
    '''

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        accuracy.reset()
        
        # training
        for images, tgt, tgt_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}: training loop"):
            images = images.to(device)
            tgt = tgt.to(device)
            tgt_mask = tgt_mask.to(device)


            outputs = model(images, tgt, tgt_mask) # forward, outputs of (batch_size, seq_len, vocab_size)
            outputs = outputs.view(-1, outputs.size(-1)) # [B * seq_len, vocab_size]
            tgt = tgt.view(-1) # [B * seq_len], (-1 simply infers dim based on other params)


            #plt.imshow(make_grid(images.cpu(), nrow=4).permute(1, 2, 0))     
            #plt.show()   
            #for t in tgt:
                 #print(tokenizer.decode(t.tolist()))    

            #greedys = model.greedy_search(images, tokenizer)

            loss = criterion(outputs, tgt)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, tgt)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        train_acc = running_acc / len(train_loader.dataset) * 100
        train_accs.append(train_acc)
        train_loss = running_loss / len(train_loader.dataset) 
        train_losses.append(train_loss)

        # valid phase
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        with torch.no_grad():
            for images, tgt, tgt_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}: val loop"):
                images = images.to(device)
                tgt = tgt.to(device)
                tgt_mask = tgt_mask.to(device)
                outputs = model(images, tgt, tgt_mask).view(-1, outputs.size(-1))
                tgt = tgt.view(-1)
                loss = criterion(outputs, tgt)
                running_loss += loss.item() * images.size(0)
                running_acc += accuracy(outputs, tgt)

            val_loss = running_loss / len(val_loader.dataset) 
            val_losses.append(val_loss)
            val_acc = running_acc / len(val_loader.dataset) * 100
            val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss} train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}")
        torch.save(model.statedict(), f"runs/Model_1{experimentNum}Epoch{epoch+1}.pt")


if __name__ == '__main__': 
    main(
        num_epochs = 3,
        model_in = Model_1(vocab_size=len(tokenizer.vocab), d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3),
        LR = 1e-4,
        dropout = 0,
        experimentNum = 1
    )
    
    



