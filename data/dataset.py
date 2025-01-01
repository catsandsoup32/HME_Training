import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch import Tensor
from PIL import Image
import os


class MathWritingDataset(Dataset): 
    def __init__(self, data_dir, cache_dir, mode=None, transform=None, tokenizer=None): # Note that data_dir is from the original data and maps to the cache
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer
        self.train_length = len(os.listdir(os.path.join(self.data_dir, 'train')))
        self.synthetic_length = len(os.listdir(os.path.join(self.data_dir, 'synthetic')))

    def __len__(self):
        if self.mode == 'train':
            return self.train_length
        elif self.mode == 'valid':
            return len(os.listdir(os.path.join(self.data_dir, 'valid')))
        elif self.mode == 'test':
            return len(os.listdir(os.path.join(self.data_dir, 'test')))
        elif self.mode == 'train+synthetic':
            return self.train_length + self.synthetic_length
        else:
            print("Typo in mode parameter")
    
    def __getitem__(self, idx):
        if self.mode != 'train+synthetic':
            fileID = str(os.listdir(os.path.join(self.data_dir, self.mode))[idx]).removesuffix('.inkml')[-16:] # Last 16 chars is the unique ID
        else:
            if idx < self.train_length: 
                fileID = str(os.listdir(os.path.join(self.data_dir, 'train'))[idx]).removesuffix('.inkml')[-16:]
            else: 
                fileID = str(os.listdir(os.path.join(self.data_dir, 'synthetic'))[idx-self.train_length]).removesuffix('.inkml')[-16:] # (idx - train_length) to start from 0

        image = Image.open(os.path.join(self.cache_dir, self.mode, fileID + '.png'))
        latex = open(os.path.join(self.cache_dir, self.mode, fileID + '.txt')).read()
        label = self.tokenizer.encode(latex)
        label = Tensor(label)
        reversed_label = torch.flip(label)    

        return self.transform(image), label, reversed_label 
    

