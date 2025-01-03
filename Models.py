import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import densenet121, DenseNet121_Weights
import math
from util import Permute, PosEncode1D, PosEncode2D


# Includes bidirectional decoder, pen stroke data in input channels, and full vocabulary
class Full_Model(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_FF, dropout, num_layers):
        super(Full_Model, self).__init__()
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT) 
        self.encoder = nn.Sequential(
            nn.Sequential(*list(densenet.children())[:-1]), 
            nn.Conv2d(1024, d_model, kernel_size=1), 
            Permute(0, 3, 2, 1), 
            PosEncode2D(d_model=d_model, dropout_percent=dropout, max_len=400, PE_temp=10000),
            nn.Flatten(1, 2)
        )
        self.tgt_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)
        self.word_PE = PosEncode1D(d_model, dropout, max_len=400, PE_temp=10000)
        self.transformer_decoder = nn.TransformerDecoder( 
            nn.TransformerDecoderLayer(d_model, nhead, dim_FF, dropout, batch_first=True),
            num_layers,
        ) 
        self.fc_out = nn.Linear(d_model, vocab_size) 
        self.d_model = d_model
    
    def decoder(self, features, tgt, tgt_mask):
        padding_mask = tgt.eq(0) 
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model) 
        tgt = self.word_PE(tgt) 
        tgt = self.transformer_decoder(tgt=tgt, memory=features, tgt_mask=tgt_mask.to(torch.float32), tgt_key_padding_mask=padding_mask.to(torch.float32), tgt_is_causal=True) 
        output = self.fc_out(tgt)
        return output
    
    def forward(self, src, tgt, reversed_tgt, tgt_mask):
        features = self.encoder(src)
        L2R_output = self.decoder(features, tgt, tgt_mask)
        R2L_output = self.decoder(features, reversed_tgt, tgt_mask)
        output = torch.stack((L2R_output, R2L_output))
        return output


class Model_1(nn.Module): 
    def __init__(self, vocab_size, d_model, nhead, dim_FF, dropout, num_layers):
        super(Model_1, self).__init__()
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT) 
        
        self.encoder = nn.Sequential(
            nn.Sequential(*list(densenet.children())[:-1]), # remove the final layer, output (B, 1024, 12, 16)
            nn.Conv2d(1024, d_model, kernel_size=1), # 1x1 convolution, output of (B, d_model, W, H) ex. (1, 256, 12, 16)
            Permute(0, 3, 2, 1), 
            PosEncode2D(d_model=d_model, dropout_percent=dropout, max_len=150, PE_temp=10000), # output (1, 16, 12, 256)
            nn.Flatten(1, 2)
        )
    
        self.tgt_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0) # simple lookup table, input indices 
        self.word_PE = PosEncode1D(d_model, dropout, max_len=150, PE_temp=10000)
        self.transformer_decoder = nn.TransformerDecoder( 
            nn.TransformerDecoderLayer(d_model, nhead, dim_FF, dropout, batch_first=True), # batch_first -> (batch, sequence, feature)
            num_layers,
        ) # input target and memory (last sequence of the encoder), then tgt_mask, memory_mask
        self.fc_out = nn.Linear(d_model, vocab_size) # y = xA^T + b, distribution over all tokens in vocabulary
        self.d_model = d_model

        self._initialize_weights()  

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # Glorot initialization for linear layers
                if m.bias is not None:
                    init.zeros_(m.bias)  # Bias initialized to zeros
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)  # Glorot initialization for conv layers
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)  # Glorot initialization for embedding layers

    def decoder(self, features, tgt, tgt_mask):
        padding_mask = tgt.eq(0) # checks where elements of tgt are equal to zero
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model) # tgt indices become embedding vectors and are scaled by sqrt of model size for stability
        tgt = self.word_PE(tgt) # adds positional encoding, size (B, seq_len, d_model)
        tgt = self.transformer_decoder(tgt=tgt, memory=features, tgt_mask=tgt_mask.to(torch.float32), tgt_key_padding_mask=padding_mask.to(torch.float32), tgt_is_causal=True) # type match
        output = self.fc_out(tgt) # size (B, seq_len, vocab_size
        return output
    
    def forward(self, src, tgt, tgt_mask):
        features = self.encoder(src)
        output = self.decoder(features, tgt, tgt_mask)
        return output
    

