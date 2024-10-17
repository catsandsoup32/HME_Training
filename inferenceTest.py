import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from Models import Model_1
from Tokenizer import LaTeXTokenizer

tokenizer = LaTeXTokenizer()
latexList = []
cache_path = Path(r'data\full_cache')
for latex_file in cache_path.glob("valid/*.txt"):
    with open(latex_file) as f:
            latexList.append(str(f.read()))
tokenizer.build_vocab(latexList) # Takes list as input to assign IDs

model = Model_1(vocab_size=217, d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3)

transform_define = transforms.Compose([
    transforms.Resize((512, 384)) ,
    transforms.ToTensor()
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

transformed_image = transform_define(Image.open('test1.png'))
features =  transformed_image.unsqueeze(0).to(device)  # Add batch dimension

src = features
max_seq_len = 200
with torch.no_grad():
    batch_size = src.size(0)
    features = model.encoder(src).detach()
    tgt = torch.ones(batch_size, 1).long().to(src.device)
    tgt_mask = torch.triu(
        torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1
    ).to(src.device)

    for i in range(1, max_seq_len):
        output = model.decoder(features, tgt, tgt_mask[:i, :i])
        next_probs = output[:, -1].log_softmax(dim=-1)
        next_chars = next_probs.argmax(dim=-1, keepdim=True)
        tgt = torch.cat((tgt, next_chars), dim=1)

    print([tokenizer.decode(seq.tolist()) for seq in tgt])