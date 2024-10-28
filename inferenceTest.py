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
vocab_dict = {}
vocab_dict = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3, '{': 4, '}': 5, '_': 6, 'x': 7, '^': 8, ')': 9, '(': 10, '=': 11, '1': 12, 'i': 13, 't': 14, '2': 15, 'a': 16, 'r': 17, 'm': 18, '-': 19, '\\frac': 20, '0': 21, 'd': 22, 'n': 23, ',': 24, '+': 25, '\\\\': 26, '\\begin': 27, '\\end': 28, '|': 29, 'k': 30, '&': 31, 'f': 32, '\\int': 33, '\\sqrt': 34, '3': 35, 'y': 36, 'p': 37, '\\hat': 38, 'A': 39, 's': 40, ']': 41, '[': 42, '\\partial': 43, 'c': 44, 'e': 45, '\\tilde': 46, '.': 47, '/': 48, '4': 49, 'X': 50, 'b': 51, 'j': 52, 'P': 53, 'v': 54, 'C': 55, 'S': 56, 'g': 57, 'u': 58, 'R': 59, 'z': 60, 'T': 61, 'l': 62, '\\prime': 63, 'E': 64, 'N': 65, '\\overline': 66, 'F': 67, 'B': 68, 'L': 69, 'V': 70, '5': 71, 'o': 72, '\\mu': 73, 'q': 74, 'I': 75, '\\cdot': 76, 'M': 77, '\\alpha': 78, '\\pi': 79, 'H': 80, 'D': 81, '\\}': 82, '\\{': 83, '6': 84, 'h': 85, '\\in': 86, 'G': 87, '\\sum': 88, '\\lambda': 89, 'K': 90, '*': 91, '\\prod': 92, '<': 93, 'w': 94, '\\theta': 95, 'Q': 96, '\\sigma': 97, ':': 98, '\\infty': 99, 'U': 100, '\\omega': 101, 'Y': 102, '\\rho': 103, 'Z': 104, '\\rangle': 105, '\\beta': 106, '7': 107, '\\rightarrow': 108, '\\gamma': 109, '\\epsilon': 110, 'O': 111, '\\underline': 112, '\\phi': 113, '\\le': 114, '\\notin': 115, '\\varphi': 116, 'W': 117, '\\delta': 118, '\\psi': 119, '8': 120, '\\nu': 121, '>': 122, '\\vec': 123, '\\langle': 124, '\\Delta': 125, 'J': 126, '\\times': 127, '\\dot': 128, '\\Omega': 129, '!': 130, '\\tau': 131, '9': 132, '\\pm': 133, '\\chi': 134, '\\approx': 135, '\\eta': 136, ';': 137, '\\nabla': 138, '\\mathbb': 139, '\\xi': 140, '\\Phi': 141, '\\ge': 142, '\\Psi': 143, '\\Sigma': 144, '\\sim': 145, '\\zeta': 146, '\\circ': 147, '\\Gamma': 148, '\\ne': 149, '\\forall': 150, '\\Lambda': 151, '\\mapsto': 152, '\\otimes': 153, '\\hbar': 154, '\\cup': 155, '\\equiv': 156, '\\kappa': 157, '\\Pi': 158, '\\oplus': 159, '\\subset': 160, '\\cap': 161, '\\bigcup': 162, '\\subseteq': 163, '\\wedge': 164, '\\cong': 165, '\\neg': 166, '\\Theta': 167, '\\dagger': 168, '\\oint': 169, '\\Rightarrow': 170, '\\aleph': 171, '\\lfloor': 172, '\\rfloor': 173, '\\backslash': 174, '\\emptyset': 175, '\\perp': 176, '\\#': 177, '\\propto': 178, '\\%': 179, '\\simeq': 180, '\\vee': 181, '?': 182, '\\ll': 183, '\\Vdash': 184, '\\Xi': 185, '\\leftarrow': 186, '\\bigcap': 187, '\\longrightarrow': 188, '\\bullet': 189, '\\exists': 190, '\\iint': 191, '\\vdash': 192, '\\iff': 193, '\\top': 194, '\\|': 195, '\\bigoplus': 196, '\\odot': 197, '\\lceil': 198, '\\rceil': 199, '\\leftrightarrow': 200, '\\models': 201, '\\supseteq': 202, '\\bigwedge': 203, '\\varsigma': 204, '\\rightleftharpoons': 205, '\\angle': 206, '\\vdots': 207, '\\Leftrightarrow': 208, '\\subsetneq': 209, '\\iota': 210, '\\gg': 211, '\\ominus': 212, '\\supset': 213, '\\Upsilon': 214, '\\triangle': 215, '\\_': 216}
print(vocab_dict)
model = Model_1(vocab_size=217, d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3)

transform_define = transforms.Compose([
    transforms.Resize((512, 384)) ,
    transforms.ToTensor()
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.load_state_dict(torch.load('runs/Exp6E1Step20000Loss=9.819655104309321Acc=0.5857142806053162.pt', map_location = device, weights_only=True))
model.eval()
model.to(device)


transformed_image = transform_define(Image.open('test2.png'))
if transformed_image[:,0,0].shape == torch.Size([4]):
     transformed_image = transformed_image[:3, :, :]
plt.imshow(transformed_image.permute(1,2,0))
plt.show()
features = transformed_image.unsqueeze(0).to(device)  # Add batch dimension
print("Transformed image shape:", transformed_image.shape)

src = features
max_seq_len = 256

softmax = nn.Softmax()

def greedy():
    with torch.no_grad():
        batch_size = 1
        tgt_in = torch.ones([1, 1], dtype=torch.long).to(device)
        tgt_mask = torch.triu(torch.ones(200, 200) * float("-inf"), diagonal=1).to(device)

        for i in range(1, 200):
            output = model(features, tgt_in, tgt_mask[:i, :i])
            print(output.shape)
            sequence_pred = torch.argmax(output, dim=-1, keepdim=True)
            tgt_in = torch.cat((tgt_in, sequence_pred[:,-1]), dim=1)
            if int(sequence_pred[:, -1]) == 2:
                 break
            print(tgt_in)
    
    print(tgt_in[0, :].shape)
    
    latex_out = tokenizer.decode(t for t in tgt_in[0, :].tolist())
    print(latex_out)


def beam_search():
    pass          


