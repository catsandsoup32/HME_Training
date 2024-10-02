import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import densenet121, DenseNet121_Weights
import math

from Util import PatchEmbedding, Permute, PosEncode1D, PosEncode2D


class Model_1(nn.Module): # from https://actamachina.com/handwritten-mathematical-expression-recognition, CNN encoder and then transformer decoder
    def __init__(self, vocab_size, d_model, nhead, dim_FF, dropout, num_layers):
        super(Model_1, self).__init__()
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            nn.Sequential(*list(densenet.children())[:-1]),  # remove the final layer
            nn.Conv2d(1024, d_model, kernel_size=1),
            Permute(0, 2, 3, 1), # reorders this from (N C H W) to be (N H W C)
            PosEncode2D(d_model=d_model, dropout_percent=dropout, max_len=30, PE_temp=10000),
            nn.Flatten(1, 2) # H and W are flattened
        ) 
    
        self.tgt_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0) # simple lookup table, input indices only
        self.word_PE = PosEncode1D(d_model, dropout, max_len=100, PE_temp=10000)
        self.transformer_decoder = nn.TransformerDecoder( 
            nn.TransformerDecoderLayer(d_model, nhead, dim_FF, dropout, batch_first=True), # batch_first -> (batch, sequence, feature)
            num_layers,
        ) # input target and memory (last sequence of the encoder), then tgt_mask, memory_mask
        self.fc_out = nn.Linear(d_model, vocab_size) # y = xA^T + b, distribution over all tokens in vocabulary
        self.d_model = d_model

    def decoder(self, features, tgt, tgt_mask):
        padding_mask = tgt.eq(0) # checks where elements of tgt are equal to zero
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model) # tgt indices become embedding vectors and are scaled by sqrt of model size for stability
        tgt = self.word_PE(tgt) # adds positional encoding
        tgt = self.transformer_decoder(tgt, features, tgt_mask=tgt_mask.to(torch.float32), tgt_key_padding_mask=padding_mask.to(torch.float32)) # type match
        output = self.fc_out(tgt)
        self.tgt_for_loss = tgt
        return output
    
    def forward(self, src, tgt, tgt_mask):
        features = self.encoder(src)
        output = self.decoder(features, tgt, tgt_mask)
        return output, self.tgt_for_loss