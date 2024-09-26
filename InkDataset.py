import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch import Tensor
from PIL import Image
import os
from Tokenizer import LaTeXTokenizer

class MathWritingDataset(Dataset): 
    def __init__(self, data_dir, cache_dir, mode=None, transform=None): # Note that data_dir is from the original data, not cache
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform

        self.tokenizer = LaTeXTokenizer()

    def __len__(self):
        if self.mode == 'train':
            return len(os.listdir(os.path.join(self.data_dir, 'train')))
        elif self.mode == 'valid':
            return len(os.listdir(os.path.join(self.data_dir, 'valid')))
        elif self.mode == 'test':
            return len(os.listdir(os.path.join(self.data_dir, 'test')))
        elif self.mode == 'train+synthetic':
            return len(os.listdir(os.path.join(self.data_dir, 'train'))) + len(os.listdir(os.path.join(self.data_dir, 'synthetic')))
    
    def __getitem__(self, idx):
        if self.mode != 'train+synthetic':
            fileID = str( os.listdir(os.path.join(self.data_dir, self.mode))[idx] ).removesuffix('.inkml')[-16:] 
            image = Image.open(os.listdir(os.path.join(self.cache_dir, self.mode, fileID + '.png'))[idx])
            
            sequence = None

            latex = open(os.listdir(os.path.join(self.cache_dir, self.mode, fileID + '.txt'))[idx]).read()
            label = self.tokenizer.encode(latex)
        else: 
            pass

        return self.transform(image), sequence, Tensor(label)
