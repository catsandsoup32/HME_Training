import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import cv2
from PIL import Image
import math

from data.dataset import MathWritingDataset
from tokenizer import LaTeXTokenizer
from models import Model_1, Full_Model


# Setup vocab, collate_fn, transform -> datasets
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = Path(r'data\mathwriting_2024')
cache_path = Path(r'data\full_cache')
tokenizer = LaTeXTokenizer()
tokenizer.id_to_token = "{0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[UNK]', 4: '{', 5: '}', 6: '_', 7: '1', 8: ')', 9: '(', 10: '^', 11: '2', 12: '=', 13: '\\frac', 14: 'x', 15: '0', 16: '-', 17: 'i', 18: ',', 19: 'n', 20: 't', 21: 'a', 22: '+', 23: '3', 24: 'r', 25: '.', 26: 'm', 27: 'd', 28: '4', 29: '5', 30: '6', 31: 'k', 32: 's', 33: '7', 34: '8', 35: '9', 36: 'c', 37: 'p', 38: 'e', 39: '|', 40: 'f', 41: 'y', 42: '\\cdot', 43: '\\sqrt', 44: 'A', 45: 'b', 46: 'l', 47: 'o', 48: '[', 49: ']', 50: 'g', 51: '\\partial', 52: 'z', 53: 'j', 54: '/', 55: 'v', 56: 'X', 57: 'T', 58: 'R', 59: '&', 60: 'u', 61: 'P', 62: '\\\\', 63: 'S', 64: 'B', 65: 'C', 66: 'E', 67: '\\prime', 68: 'N', 69: '\\begin', 70: '\\end', 71: '\\alpha', 72: '\\int', 73: 'F', 74: 'V', 75: 'q', 76: '\\pi', 77: 'L', 78: 'h', 79: '\\sum', 80: 'M', 81: '\\theta', 82: '\\mu', 83: '\\in', 84: 'H', 85: 'I', 86: '\\hat', 87: 'D', 88: '\\rightarrow', 89: 'G', 90: '\\lambda', 91: '*', 92: '\\sigma', 93: '\\{', 94: ':', 95: '\\}', 96: 'w', 97: '\\overline', 98: '\\infty', 99: '\\mathbb', 100: 'K', 101: '\\beta', 102: '\\omega', 103: '\\rho', 104: '\\epsilon', 105: 'Z', 106: 'Q', 107: 'Y', 108: '\\le', 109: '\\gamma', 110: 'U', 111: '\\times', 112: '\\phi', 113: '\\rangle', 114: '\\Delta', 115: '<', 116: '\\tilde', 117: '\\psi', 118: '\\delta', 119: '\\nu', 120: 'W', 121: '\\tau', 122: 'O', 123: '\\varphi', 124: 'J', 125: '\\langle', 126: '\\vec', 127: '!', 128: '>', 129: '\\nabla', 130: '\\ge', 131: '\\prod', 132: '\\Omega', 133: ';', 134: '\\eta', 135: '\\Gamma', 136: '\\approx', 137: '\\xi', 138: '\\Phi', 139: '\\dot', 140: '\\pm', 141: '\\otimes', 142: '\\circ', 143: '\\wedge', 144: '\\equiv', 145: '\\hbar', 146: '\\chi', 147: '\\underline', 148: '\\zeta', 149: '\\kappa', 150: '\\ne', 151: '\\forall', 152: '\\Sigma', 153: '\\sim', 154: '\\subseteq', 155: '\\Psi', 156: '\\notin', 157: '\\cap', 158: '\\Lambda', 159: '\\mapsto', 160: '\\neg', 161: '\\cup', 162: '\\oplus', 163: '\\Rightarrow', 164: '\\dagger', 165: '\\vee', 166: '\\subset', 167: '\\backslash', 168: '\\rfloor', 169: '\\Pi', 170: '\\lfloor', 171: '\\|', 172: '\\Theta', 173: '\\exists', 174: '\\cong', 175: '\\emptyset', 176: '\\propto', 177: '\\perp', 178: '\\vdash', 179: '\\bigcup', 180: '\\bullet', 181: '\\simeq', 182: '\\leftarrow', 183: '\\aleph', 184: '\\%', 185: '\\vartheta', 186: '\\ll', 187: '\\#', 188: '\\oint', 189: '\\angle', 190: '\\top', 191: '\\leftrightarrow', 192: '\\bigoplus', 193: '\\iint', 194: '\\bigcap', 195: '\\vdots', 196: '\\lceil', 197: '\\rceil', 198: '\\iff', 199: '\\gg', 200: '\\odot', 201: '\\varpi', 202: '\\Leftrightarrow', 203: '\\models', 204: '\\longrightarrow', 205: '\\ominus', 206: '\\iota', 207: '?', 208: '\\upsilon', 209: '\\mp', 210: '\\Xi', 211: '\\varsigma', 212: '\\hookrightarrow', 213: '\\supseteq', 214: '\\supset', 215: '\\subsetneq', 216: '\\triangleq', 217: '\\bigwedge', 218: '\\div', 219: '\\Upsilon', 220: '\\Vdash', 221: '\\&', 222: '\\bigvee', 223: '\\ni', 224: '\\rightleftharpoons', 225: '\\triangle', 226: '\\_', 227: '\\not', 228: '\\bigcirc', 229: '\\sqsubseteq', 230: '\\triangleleft'}"
tokenizer.token_to_id = "{'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3, '{': 4, '}': 5, '_': 6, '1': 7, ')': 8, '(': 9, '^': 10, '2': 11, '=': 12, '\\frac': 13, 'x': 14, '0': 15, '-': 16, 'i': 17, ',': 18, 'n': 19, 't': 20, 'a': 21, '+': 22, '3': 23, 'r': 24, '.': 25, 'm': 26, 'd': 27, '4': 28, '5': 29, '6': 30, 'k': 31, 's': 32, '7': 33, '8': 34, '9': 35, 'c': 36, 'p': 37, 'e': 38, '|': 39, 'f': 40, 'y': 41, '\\cdot': 42, '\\sqrt': 43, 'A': 44, 'b': 45, 'l': 46, 'o': 47, '[': 48, ']': 49, 'g': 50, '\\partial': 51, 'z': 52, 'j': 53, '/': 54, 'v': 55, 'X': 56, 'T': 57, 'R': 58, '&': 59, 'u': 60, 'P': 61, '\\\\': 62, 'S': 63, 'B': 64, 'C': 65, 'E': 66, '\\prime': 67, 'N': 68, '\\begin': 69, '\\end': 70, '\\alpha': 71, '\\int': 72, 'F': 73, 'V': 74, 'q': 75, '\\pi': 76, 'L': 77, 'h': 78, '\\sum': 79, 'M': 80, '\\theta': 81, '\\mu': 82, '\\in': 83, 'H': 84, 'I': 85, '\\hat': 86, 'D': 87, '\\rightarrow': 88, 'G': 89, '\\lambda': 90, '*': 91, '\\sigma': 92, '\\{': 93, ':': 94, '\\}': 95, 'w': 96, '\\overline': 97, '\\infty': 98, '\\mathbb': 99, 'K': 100, '\\beta': 101, '\\omega': 102, '\\rho': 103, '\\epsilon': 104, 'Z': 105, 'Q': 106, 'Y': 107, '\\le': 108, '\\gamma': 109, 'U': 110, '\\times': 111, '\\phi': 112, '\\rangle': 113, '\\Delta': 114, '<': 115, '\\tilde': 116, '\\psi': 117, '\\delta': 118, '\\nu': 119, 'W': 120, '\\tau': 121, 'O': 122, '\\varphi': 123, 'J': 124, '\\langle': 125, '\\vec': 126, '!': 127, '>': 128, '\\nabla': 129, '\\ge': 130, '\\prod': 131, '\\Omega': 132, ';': 133, '\\eta': 134, '\\Gamma': 135, '\\approx': 136, '\\xi': 137, '\\Phi': 138, '\\dot': 139, '\\pm': 140, '\\otimes': 141, '\\circ': 142, '\\wedge': 143, '\\equiv': 144, '\\hbar': 145, '\\chi': 146, '\\underline': 147, '\\zeta': 148, '\\kappa': 149, '\\ne': 150, '\\forall': 151, '\\Sigma': 152, '\\sim': 153, '\\subseteq': 154, '\\Psi': 155, '\\notin': 156, '\\cap': 157, '\\Lambda': 158, '\\mapsto': 159, '\\neg': 160, '\\cup': 161, '\\oplus': 162, '\\Rightarrow': 163, '\\dagger': 164, '\\vee': 165, '\\subset': 166, '\\backslash': 167, '\\rfloor': 168, '\\Pi': 169, '\\lfloor': 170, '\\|': 171, '\\Theta': 172, '\\exists': 173, '\\cong': 174, '\\emptyset': 175, '\\propto': 176, '\\perp': 177, '\\vdash': 178, '\\bigcup': 179, '\\bullet': 180, '\\simeq': 181, '\\leftarrow': 182, '\\aleph': 183, '\\%': 184, '\\vartheta': 185, '\\ll': 186, '\\#': 187, '\\oint': 188, '\\angle': 189, '\\top': 190, '\\leftrightarrow': 191, '\\bigoplus': 192, '\\iint': 193, '\\bigcap': 194, '\\vdots': 195, '\\lceil': 196, '\\rceil': 197, '\\iff': 198, '\\gg': 199, '\\odot': 200, '\\varpi': 201, '\\Leftrightarrow': 202, '\\models': 203, '\\longrightarrow': 204, '\\ominus': 205, '\\iota': 206, '?': 207, '\\upsilon': 208, '\\mp': 209, '\\Xi': 210, '\\varsigma': 211, '\\hookrightarrow': 212, '\\supseteq': 213, '\\supset': 214, '\\subsetneq': 215, '\\triangleq': 216, '\\bigwedge': 217, '\\div': 218, '\\Upsilon': 219, '\\Vdash': 220, '\\&': 221, '\\bigvee': 222, '\\ni': 223, '\\rightleftharpoons': 224, '\\triangle': 225, '\\_': 226, '\\not': 227, '\\bigcirc': 228, '\\sqsubseteq': 229, '\\triangleleft': 230}"

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

train_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='train', transform=transform, tokenizer=tokenizer)
valid_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='valid', transform=transform, tokenizer=tokenizer)
test_dataset = MathWritingDataset(data_dir=data_path, cache_dir=cache_path, mode='test', transform=transform, tokenizer=tokenizer)


def main(num_epochs, model_in, lr, experiment_num, use_test_in_train):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8 , collate_fn=collate_fn) 
    val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn)

    accuracy = Accuracy(task='multiclass', num_classes=len(tokenizer.vocab))
    accuracy = accuracy.to(device)
    model = model_in
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-6, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    train_accs, train_losses, val_accs, val_losses = [], [], [], []

    #model.load_state_dict(torch.load("runs/Exp8E15End_Acc=0.6744208335876465.pt", map_location=device, weights_only=True))
    # change to range(x, num_epochs) if loading from saved
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        accuracy.reset()
        counter = 0 
        
        for images, tgt, reversed_tgt, tgt_mask in tqdm(train_loader, desc=f" Epoch {epoch} Training Loop"):
            counter += 1
            images = images.to(device)
            tgt = tgt.to(device)
            reversed_tgt = reversed_tgt.to(device)
            tgt_mask = tgt_mask.to(device)

            # Show sample images
            # plt.imshow(make_grid(images.cpu(), nrow=4).permute(1, 2, 0))
            # plt.show()
            # plt.axis("off")

            tgt_in, reversed_tgt_in = tgt[:, :-1], reversed_tgt[:, :-1]
            tgt_out, reversed_tgt_out = tgt[:, 1:], reversed_tgt[:, 1:] # So it doesn't learn to just copy but to actually predict the next token
            outputs = model(images, tgt_in, reversed_tgt_in, tgt_mask[:-1, :-1]) # forward, outputs of (batch_size, seq_len, vocab_size)
            outputs_reshaped_L, outputs_reshaped_R = outputs[0].view(-1, outputs[0].size(-1)), outputs[1].view(-1, outputs[1].size(-1)) # [B * seq_len, vocab_size]

            loss_L = criterion(outputs_reshaped_L, tgt_out.reshape(-1))
            loss_R = criterion(outputs_reshaped_R, reversed_tgt_out.reshape(-1))
            loss = 0.5*loss_L + 0.5*loss_R
            running_loss += loss.item() * images.size(0)
            running_acc += (0.5 * accuracy(outputs[0].permute(0, 2, 1), tgt_out) + 0.5 * accuracy(outputs[1].permute(0, 2, 1), reversed_tgt_out))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # This gives more unique data to train on
        if use_test_in_train:
            for images, tgt, reversed_tgt, tgt_mask in tqdm(test_loader, desc=f" Epoch {epoch} Training Loop (Test Samples)"):
                counter += 1
                images = images.to(device)
                tgt = tgt.to(device)
                reversed_tgt = reversed_tgt.to(device)
                tgt_mask = tgt_mask.to(device)
                tgt_in, reversed_tgt_in = tgt[:, :-1], reversed_tgt[:, :-1]
                tgt_out, reversed_tgt_out = tgt[:, 1:], reversed_tgt[:, 1:] # So it doesn't learn to just copy but to actually predict the next token
                outputs = model(images, tgt_in, reversed_tgt_in, tgt_mask[:-1, :-1]) # forward, outputs of (batch_size, seq_len, vocab_size)
                outputs_reshaped_L, outputs_reshaped_R = outputs[0].view(-1, outputs[0].size(-1)), outputs[1].view(-1, outputs[1].size(-1)) # [B * seq_len, vocab_size]
                loss_L = criterion(outputs_reshaped_L, tgt_out.reshape(-1))
                loss_R = criterion(outputs_reshaped_R, reversed_tgt_out.reshape(-1))
                loss = 0.5*loss_L + 0.5*loss_R
                running_loss += loss.item() * images.size(0)
                running_acc += (0.5 * accuracy(outputs[0].permute(0, 2, 1), tgt_out) + 0.5 * accuracy(outputs[1].permute(0, 2, 1), reversed_tgt_out))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        train_acc = running_acc / counter * 100
        train_accs.append(train_acc)
        train_loss = running_loss / counter
        train_losses.append(train_loss)
        print(f"Training: loss = {train_loss}, acc = {train_acc}")


        # Valid Loop
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        counter = 0
        with torch.no_grad():
            for images, tgt, reversed_tgt, tgt_mask in tqdm(val_loader, desc=f"Epoch {epoch+1}: val loop"):
                counter += 1
                images = images.to(device)
                tgt = tgt.to(device)
                reversed_tgt = tgt.to(device)
                tgt_mask = tgt_mask.to(device)
                tgt_in, reversed_tgt_in = tgt[:, :-1], reversed_tgt[:, :-1]
                tgt_out, reversed_tgt_out = tgt[:, 1:], reversed_tgt[:, 1:]
                outputs = model(images, tgt_in, reversed_tgt_in, tgt_mask[:-1, :-1]) # forward, outputs of (batch_size, seq_len, vocab_size)
                outputs_reshaped_L, outputs_reshaped_R = outputs[0].view(-1, outputs[0].size(-1)), outputs[1].view(-1, outputs[1].size(-1))
                loss = 0.5 * criterion(outputs_reshaped_L, tgt_out.reshape(-1)) + criterion(outputs_reshaped_R, tgt_out.reshape(-1))
                running_loss += loss.item() * images.size(0)
                running_acc += 0.5 * accuracy(outputs[0].permute(0, 2, 1), tgt_out) + 0.5 * accuracy(outputs[1].permute(0, 2, 1), reversed_tgt_out)

        val_loss = running_loss / counter
        val_losses.append(val_loss)
        val_acc = running_acc / counter * 100
        val_accs.append(val_acc)
        print(f"Validation: loss = {val_loss}, acc = {val_acc}")
        scheduler.step(val_loss)

        torch.save(model.state_dict(), f"runs/Exp{experiment_num}E{epoch}.pt")
    
    # Generate graphs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Losses', marker='o')  
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Losses', marker='s')  
    plt.legend()
    plt.savefig(f"runs/Exp{experiment_num}Losses")
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Accs', marker='o')  
    plt.plot(range(1, num_epochs+1), val_accs, label='Val Accs', marker='s')  
    plt.legend()
    plt.savefig(f"runs/Exp{experiment_num}Accs")
    plt.close()


if __name__ == '__main__': 
    main(
        num_epochs = 20,
        model_in = Full_Model(vocab_size=len(tokenizer.vocab), d_model=256, nhead=8, dim_FF=1024, dropout=0.3, num_layers=3),
        lr = 1e-4,
        experiment_num = 9,
        use_test_in_train = True
    )
    
# Experiment 5 should have fixed attention mask (didn't implement teacher forcing right, as loss was comparing sequences with BOS in it)

# Experiment 6 is with no pretrained weights

# Experiment 7 has pretrained weights and Adadelta optimizer
# optimizer = optim.Adadelta(model.parameters(), lr=LR, rho=0.9, eps=1e-06, weight_decay=0)
# why so bad :'(  -- could be because set LR for Adadelta ... 
 
# Experiment 8 uses pretrained and Adam, also implemented thickness transform and plan to train on more epochs
# optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-4)

# Experiment 9 adds color channel embedding and bidirectionality
