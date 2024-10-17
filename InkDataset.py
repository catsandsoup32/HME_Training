import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch import Tensor
from PIL import Image
import os
from Tokenizer import LaTeXTokenizer
from pathlib import Path

data_path = Path(r'data\mathwriting_2024')
cache_path = Path(r'data\full_cache')

tokenizer = LaTeXTokenizer()
latexList = []
for latex_file in cache_path.glob("valid/*.txt"):
    with open(latex_file) as f:
            latexList.append(str(f.read()))

class MathWritingDataset(Dataset): 
    def __init__(self, data_dir, cache_dir, mode=None, transform=None): # Note that data_dir is from the original data, not cache
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform
        tokenizer.build_vocab(latexList) # Takes list as input to assign IDs

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
        fileID = str( os.listdir(os.path.join(self.data_dir, self.mode))[idx] ).removesuffix('.inkml')[-16:] 
        image = Image.open(os.path.join(self.cache_dir, self.mode, fileID + '.png'))
        
        sequence = None

        latex = open(os.path.join(self.cache_dir, self.mode, fileID + '.txt')).read()
        label = tokenizer.encode(latex)
        #print(f"Dataset vocab: {tokenizer.vocab}")
        #print(f"Dataset label: {label}")
    

        #return self.transform(image), sequence, label
        return self.transform(image), Tensor(label)

# train_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='train', transform=None)
