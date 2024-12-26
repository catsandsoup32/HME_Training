import torch
import torch.nn as nn
from torch import Tensor 


class Permute(nn.Module):
    def __init__(self, *dims: int): 
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims) # reorders the tuple
    
    
class PosEncode1D(nn.Module):
    def __init__(self, d_model, dropout_percent, max_len, PE_temp):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1) # creates a vector (max_len x 1), 1 is needed for matmul operations
        dim_t = torch.arange(0, d_model, 2) # 2i term in the denominator exponent
        scaling = PE_temp **(dim_t/d_model) # entire denominator 

        pe = torch.zeros(max_len, d_model) # 
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
            pe_2D[:, :, i] = pe[:, i].unsqueeze(1) + pe[:, i].unsqueeze(0)  # first unsqueeze changed from -1

        self.dropout = nn.Dropout(dropout_percent)
        self.register_buffer("pe", pe_2D) 

    def forward(self, x):
        batch, height, width, d_model = x.shape
        return self.dropout(x + self.pe[None, :height, :width, :])


