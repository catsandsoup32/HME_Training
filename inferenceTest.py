import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch import log_softmax

from Models import Model_1
from Tokenizer import LaTeXTokenizer

tokenizer = LaTeXTokenizer()
latexList = []
'''
cache_path = Path(r'data\full_cache')
for latex_file in cache_path.glob("valid/*.txt"):
    with open(latex_file) as f:
            latexList.append(str(f.read()))
tokenizer.build_vocab(latexList) # Takes list as input to assign IDs
'''
tokenizer.id_to_token = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[UNK]', 4: '{', 5: '}', 6: '_', 7: 'x', 8: '^', 9: ')', 10: '(', 11: '=', 12: '1', 13: 'i', 14: 't', 15: '2', 16: 'a', 17: 'r', 18: 'm', 19: '-', 20: '\\frac', 21: '0', 22: 'd', 23: 'n', 24: ',', 25: '+', 26: '\\\\', 27: '\\begin', 28: '\\end', 29: '|', 30: 'k', 31: '&', 32: 'f', 33: '\\int', 34: '\\sqrt', 35: '3', 36: 'y', 37: 'p', 38: '\\hat', 39: 'A', 40: 's', 41: ']', 42: '[', 43: '\\partial', 44: 'c', 45: 'e', 46: '\\tilde', 47: '.', 48: '/', 49: '4', 50: 'X', 51: 'b', 52: 'j', 53: 'P', 54: 'v', 55: 'C', 56: 'S', 57: 'g', 58: 'u', 59: 'R', 60: 'z', 61: 'T', 62: 'l', 63: '\\prime', 64: 'E', 65: 'N', 66: '\\overline', 67: 'F', 68: 'B', 69: 'L', 70: 'V', 71: '5', 72: 'o', 73: '\\mu', 74: 'q', 75: 'I', 76: '\\cdot', 77: 'M', 78: '\\alpha', 79: '\\pi', 80: 'H', 81: 'D', 82: '\\}', 83: '\\{', 84: '6', 85: 'h', 86: '\\in', 87: 'G', 88: '\\sum', 89: '\\lambda', 90: 'K', 91: '*', 92: '\\prod', 93: '<', 94: 'w', 95: '\\theta', 96: 'Q', 97: '\\sigma', 98: ':', 99: '\\infty', 100: 'U', 101: '\\omega', 102: 'Y', 103: '\\rho', 104: 'Z', 105: '\\rangle', 106: '\\beta', 107: '7', 108: '\\rightarrow', 109: '\\gamma', 110: '\\epsilon', 111: 'O', 112: '\\underline', 113: '\\phi', 114: '\\le', 115: '\\notin', 116: '\\varphi', 117: 'W', 118: '\\delta', 119: '\\psi', 120: '8', 121: '\\nu', 122: '>', 123: '\\vec', 124: '\\langle', 125: '\\Delta', 126: 'J', 127: '\\times', 128: '\\dot', 129: '\\Omega', 130: '!', 131: '\\tau', 132: '9', 133: '\\pm', 134: '\\chi', 135: '\\approx', 136: '\\eta', 137: ';', 138: '\\nabla', 139: '\\mathbb', 140: '\\xi', 141: '\\Phi', 142: '\\ge', 143: '\\Psi', 144: '\\Sigma', 145: '\\sim', 146: '\\zeta', 147: '\\circ', 148: '\\Gamma', 149: '\\ne', 150: '\\forall', 151: '\\Lambda', 152: '\\mapsto', 153: '\\otimes', 154: '\\hbar', 155: '\\cup', 156: '\\equiv', 157: '\\kappa', 158: '\\Pi', 159: '\\oplus', 160: '\\subset', 161: '\\cap', 162: '\\bigcup', 163: '\\subseteq', 164: '\\wedge', 165: '\\cong', 166: '\\neg', 167: '\\Theta', 168: '\\dagger', 169: '\\oint', 170: '\\Rightarrow', 171: '\\aleph', 172: '\\lfloor', 173: '\\rfloor', 174: '\\backslash', 175: '\\emptyset', 176: '\\perp', 177: '\\#', 178: '\\propto', 179: '\\%', 180: '\\simeq', 181: '\\vee', 182: '?', 183: '\\ll', 184: '\\Vdash', 185: '\\Xi', 186: '\\leftarrow', 187: '\\bigcap', 188: '\\longrightarrow', 189: '\\bullet', 190: '\\exists', 191: '\\iint', 192: '\\vdash', 193: '\\iff', 194: '\\top', 195: '\\|', 196: '\\bigoplus', 197: '\\odot', 198: '\\lceil', 199: '\\rceil', 200: '\\leftrightarrow', 201: '\\models', 202: '\\supseteq', 203: '\\bigwedge', 204: '\\varsigma', 205: '\\rightleftharpoons', 206: '\\angle', 207: '\\vdots', 208: '\\Leftrightarrow', 209: '\\subsetneq', 210: '\\iota', 211: '\\gg', 212: '\\ominus', 213: '\\supset', 214: '\\Upsilon', 215: '\\triangle', 216: '\\_'}

model = Model_1(vocab_size=217, d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3)

transform_define = transforms.Compose([
    transforms.Resize((512, 384)) ,
    transforms.ToTensor()
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.load_state_dict(torch.load('runs/Exp8E8End_Acc=0.6702922582626343.pt', map_location = device, weights_only=True))
model.eval()
model.to(device)


transformed_image = transform_define(Image.open('test3.png'))
if transformed_image[:,0,0].shape == torch.Size([4]):
     transformed_image = transformed_image[:3, :, :]
plt.imshow(transformed_image.permute(1,2,0))
#plt.show()
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


def beam_search(width, alpha):
    tgt_in = torch.ones([1, 1], dtype=torch.long).to(device)
    tgt_mask = torch.triu(torch.ones(200, 200) * float("-inf"), diagonal=1).to(device)
    
    output = model(features, tgt_in, tgt_mask[:1, :1])
    output_sm = log_softmax(output, dim=-1) # Size (1, 1, 217)
    top_probs, indices = output_sm.topk(width, dim=-1)
    top_probs, indices = top_probs.tolist(), indices.tolist()

    for i in range(width):
        pass
#beam_search(2, None)

greedy()