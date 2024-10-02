import torch
import torch.nn as nn
from torch import Tensor
import math

class PatchEmbedding(nn.Module):
    # Images are (512, 384), 768 patches in total
    def __init__(self, in_channels=1, patch_size=16, emb_size=768, img_size=(512, 384)): 
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size) # Gets patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, emb_size, num_patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, emb_size)
        x = x + self.pos_embed
        return x 
    
class Permute(nn.Module):
    def __init__(self, *dims: int): # asterik accepts arbitary amount of arguments
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims) # reorders the tuple
    

class PosEncode1D(nn.Module):
    def __init__(self, d_model, dropout_percent, max_len, PE_temp):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1) # creates a vector (max_len x 1), 1 is needed for matmul operations
        dim_t = torch.arange(0, d_model, 2)
        scaling = PE_temp **(dim_t/d_model)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position / scaling) # every second term starting from 0 (even)
        pe[:, 1::2] = torch.cos(position / scaling) # every second term starting from 1 (odd)

        self.dropout = nn.Dropout(dropout_percent)
        self.register_buffer("pe", pe) # stores pe tensor to be used but not updated

    def forward(self, x):
        batch, sequence_length, d_model = x.shape 
        return self.dropout(x + self.pe[None, :sequence_length, :]) # None to broadcast across batch, adds element-wise [x + pe, . . .]


class PosEncode2D(nn.Module):
    def __init__(self, d_model, dropout_percent, max_len, PE_temp):
        super().__init__()
        # 1D encoding
        position = torch.arange(max_len).unsqueeze(1) 
        dim_t = torch.arange(0, d_model, 2)
        scaling = PE_temp **(dim_t/d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position / scaling)
        pe[:, 1::2] = torch.cos(position / scaling)

        pe_2D = torch.zeros(max_len, max_len, d_model) # some outer product magic
        for i in range(d_model):
            pe_2D[:, :, i] = pe[:, i].unsqueeze(-1) + pe[:, i].unsqueeze(0) 

        self.dropout = nn.Dropout(dropout_percent)
        self.register_buffer("pe", pe_2D) 

    def forward(self, x):
        batch, height, width, d_model = x.shape
        return self.dropout(x + self.pe[None, :height, :width, :])



