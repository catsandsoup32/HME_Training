import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer import LaTeXTokenizer
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
    
    def __getitem__(self, idx):
        fileID = str( os.listdir(os.path.join(self.data_dir, self.mode))[idx] ).removesuffix('.inkml')[-16:] 
        image = Image.open(os.path.join(self.cache_dir, self.mode, fileID + '.png'))
        
        latex = open(os.path.join(self.cache_dir, self.mode, fileID + '.txt')).read()
        label = tokenizer.encode(latex)
        label = Tensor(label)
        reversed_label = torch.flip(label, dims=[0])    

        return self.transform(image), label, reversed_label 
    

