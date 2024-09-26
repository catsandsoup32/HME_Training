import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import densenet121, DenseNet121_Weights

from HME_Training.Util import PatchEmbedding, Permute, PosEncode1D, PosEncode2D


class Model_1(nn.Module): # from https://actamachina.com/handwritten-mathematical-expression-recognition, CNN encoder and then transformer decoder
    def __init__(self, vocab_size, d_model, nhead, dim_FF, dropout, num_layers):
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            nn.Sequential(*list(densenet.children())[:-1]),  # remove the final layer
            nn.Conv2d(1024, d_model, kernel_size=1),
            Permute(0, 2, 3, 1), # reorders this from (N C H W) to be (N H W C)
            PosEncode2D(d_model, dropout, max_len=30, PE_temp=10000),
            nn.Flatten(1, 2) # H and W are flattened
        ) 
    
        self.tgt_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0) # simple lookup table, input indices only
        self.word_PE = PosEncode1D(d_model, dropout)
        self.transformer_decoder = nn.TransformerDecoder( 
            nn.TransformerDecoderLayer(d_model, nhead, dim_FF, dropout, batch_first=True), # batch first becomes (batch, sequence, feature)
            num_layers,
        ) # input target and memory (last sequence of the encoder), then tgt_mask, memory_mask
        self.fc_out = nn.Linear(d_model, vocab_size) # y = xA^T + b, distribution over all tokens in vocabulary

    def decoder(self, features, tgt, tgt_mask):
        padding_mask = tgt.eq(0) # checks where elements of tgt are equal to zero

        pass
    


    def forward(self):
        pass