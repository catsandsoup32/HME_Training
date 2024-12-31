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
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
import cv2
from PIL import Image
import math

from data.dataset import MathWritingDataset
from tokenizer import LaTeXTokenizer

from models import Model_1

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


class thicknessTransform(nn.Module):
    def forward(self, img):
        enable = np.random.randint(0,1)
        random_num = np.random.randint(1, 5)
        kernel = np.ones((random_num, random_num), np.uint8) 
        img_np = np.array(img)
        if enable:
            transform_img = cv2.erode(img_np, kernel)
        else:
            random_num = np.random.randint(1,3)
            transform_img = cv2.dilate(img_np, np.ones((random_num, random_num), np.uint8))
        return Image.fromarray(transform_img)


def collate_fn(batch):
    images, labels, reversed_labels = zip(*batch)
    tgt = pad_sequence(labels, batch_first=True).long() # Pads all sequences to the max of the batch
    reversed_tgt = pad_sequence(reversed_labels, batch_first=True).long()
    seq_len = tgt.size(1) # Max sequence length - because tgt is of size (batch x max_length)
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float32) * float("-inf"), diagonal=1) # ATTENTION MASK, see https://pytorch.org/docs/stable/generated/torch.triu.html
    images = torch.stack(images)  # tensor of shape (batch_size, C, H, W)
    return images, tgt, reversed_tgt, tgt_mask


transform = transforms.Compose([
    thicknessTransform(),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
    transforms.ToTensor()
])

train_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='train', transform=transform)
valid_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='valid', transform=transform)
test_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='test', transform=transform)


def main(num_epochs, model_in, LR, experimentNum):
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8 , collate_fn=collate_fn) 
    val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    accuracy = Accuracy(task='multiclass', num_classes=len(tokenizer.vocab))
    accuracy = accuracy.to(device)
    model = model_in
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-6, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    train_accs, train_losses, val_accs, val_losses = [], [], [], []
    
    loss_over_epoch = []

    #model.load_state_dict(torch.load("runs/Exp8E15End_Acc=0.6744208335876465.pt", map_location=device, weights_only=True))
    # change to range(1, num_epochs) if loading from saved
    for epoch in range(0, num_epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        accuracy.reset()
        counter = 0 
        
        # training
        for images, tgt, reversed_tgt, tgt_mask in tqdm(train_loader, desc=f" Epoch {epoch+1}: training loop"):
            images = images.to(device)
            tgt = tgt.to(device)
            reversed_tgt = reversed_tgt.to(device)
            tgt_mask = tgt_mask.to(device)
            counter += 1
            torch.set_printoptions(threshold=None)

            tgt_in, reversed_tgt_in = tgt[:, :-1], reversed_tgt[:, :-1]
            tgt_out, reversed_tgt_out = tgt[:, 1:], reversed_tgt[:, 1:] # So it doesn't learn to just copy but to actually predict the next token
           
            plt.imshow(make_grid(images.cpu(), nrow=4).permute(1, 2, 0))
            plt.show()
            plt.axis("off")
        
            print(images.shape)
            print(tgt_in.shape)
            print(tgt_mask[:-1, :-1].shape)

            outputs = model(images, tgt_in, reversed_tgt_in, tgt_mask[:-1, :-1]) # forward, outputs of (batch_size, seq_len, vocab_size)
            outputs_reshaped_L, outputs_reshaped_R = outputs[0].view(-1, outputs.size(-1)), outputs[1].view(-1, outputs.size(-1)) # [B * seq_len, vocab_size]

            loss_L = criterion(outputs_reshaped_L, tgt_out.reshape(-1))
            loss_R = criterion(outputs_reshaped_R, reversed_tgt_out.reshape(-1))
            loss = 0.5*loss_L + 0.5*loss_R

            running_loss += loss.item() * images.size(0)
            running_acc += (0.5 * accuracy(outputs[0].permute(0, 2, 1), tgt_out) + 0.5 * accuracy(outputs[1].permute(0, 2, 1), reversed_tgt_out))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if counter%15000 == 0:
                loss_over_epoch.append(str(running_loss/counter))
                print("   Saved!")
                print("   Sanity check!")
                print(f"   Target: {tgt_out[0,:]}")
                print(tgt_out.shape)
                print(f"   Pred: {torch.argmax(outputs[0,:,:], dim=-1)}")
                print(torch.argmax(outputs, dim=-1).shape)
                torch.save(model.state_dict(), f"runs/Exp{experimentNum}E{epoch+1}Step{counter}Loss={round(loss.item(), 2)}Acc={round(accuracy(outputs.permute(0,2,1),tgt_out).item()*100, 2)}.pt")

    
        train_acc = running_acc / len(train_loader.dataset) * 10
        train_accs.append(train_acc)
        train_loss = running_loss / len(train_loader.dataset) 
        train_losses.append(train_loss)

        torch.save(model.state_dict(), f"runs/Exp{experimentNum}E{epoch+1}End_Acc={train_acc}.pt")
        print(f"Loss over epoch: {loss_over_epoch}")

        # valid phase
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        with torch.no_grad():
            for images, tgt, tgt_mask in tqdm(val_loader, desc=f"Epoch {epoch+1}: val loop"):
                images = images.to(device)
                tgt = tgt.to(device)
                tgt_mask = tgt_mask.to(device)
                tgt_in = tgt[:, :-1] 
                tgt_out = tgt[:, 1:]
                outputs = model(images, tgt_in, tgt_mask[:-1, :-1]) # forward, outputs of (batch_size, seq_len, vocab_size)
                outputs_reshaped = outputs.view(-1, outputs.size(-1)) # [B * seq_len, vocab_size]
                loss = criterion(outputs_reshaped, tgt_out.reshape(-1))
                running_loss += loss.item() * images.size(0)
                running_acc += accuracy(outputs.permute(0, 2, 1), tgt_out)

        val_loss = running_loss / len(val_loader.dataset) 
        val_losses.append(val_loss)
        val_acc = running_acc / len(val_loader.dataset) * 100
        val_accs.append(val_acc)
        print(f"val loss: {val_loss}, val acc: {val_acc}")
        scheduler.step(val_loss)


if __name__ == '__main__': 
    main(
        num_epochs = 30,
        model_in = Model_1(vocab_size=len(tokenizer.vocab), d_model=256, nhead=8, dim_FF=1024, dropout=0.3, num_layers=3),
        LR = 1e-4,
        experimentNum = 9
    )
    
# Experiment 5 should have fixed attention mask (didn't implement teacher forcing right, as loss was comparing sequences with BOS in it)

# Experiment 6 is with no pretrained weights

# Experiment 7 has pretrained weights and Adadelta optimizer
# optimizer = optim.Adadelta(model.parameters(), lr=LR, rho=0.9, eps=1e-06, weight_decay=0)
# why so bad :'(  -- could be because set LR for Adadelta ... 
 
# Experiment 8 uses pretrained and Adam, also implemented thickness transform and plan to train on more epochs
# optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-4)
