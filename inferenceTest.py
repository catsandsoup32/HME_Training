import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor

from Models import Model_1
from Tokenizer import LaTeXTokenizer

tokenizer = LaTeXTokenizer()
latexList = []
cache_path = Path(r'data\full_cache')
for latex_file in cache_path.glob("valid/*.txt"):
    with open(latex_file) as f:
            latexList.append(str(f.read()))
tokenizer.build_vocab(latexList) # Takes list as input to assign IDs
vocab_dict = tokenizer.vocab
print(vocab_dict)
model = Model_1(vocab_size=217, d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3)

transform_define = transforms.Compose([
    transforms.Resize((512, 384)) ,
    transforms.ToTensor()
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.load_state_dict(torch.load('runs/Exp6E1Step28000Loss=8.171539155870676Acc=0.5581395626068115.pt', map_location = device, weights_only=True))
model.eval()
model.to(device)

transformed_image = transform_define(Image.open('test1.png'))
plt.imshow(transformed_image.permute(1,2,0))
features = transformed_image.unsqueeze(0).to(device)  # Add batch dimension
print("Transformed image shape:", transformed_image.shape)

src = features
max_seq_len = 256

softmax = nn.Softmax()

with torch.no_grad():
    batch_size = 1
    tgt_in = torch.ones([1, 1], dtype=torch.long).to(device)
    tgt_mask = torch.triu(torch.ones(200, 200) * float("-inf"), diagonal=1).to(device)


    for i in range(1, 200):
        output = model(features, tgt_in, tgt_mask[:i, :i])
        print(output.shape)
        sequence_pred = torch.argmax(output, dim=-1, keepdim=True)
        tgt_in = torch.cat((tgt_in, sequence_pred[:,-1]), dim=1)
        print(tgt_in)

