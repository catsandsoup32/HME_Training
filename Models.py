import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax
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
            PosEncode2D(d_model=d_model, dropout_percent=dropout, max_len=256, PE_temp=10000),
            nn.Flatten(1, 2) # H and W are flattened
        ) 
    
        self.tgt_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0) # simple lookup table, input indices only
        self.word_PE = PosEncode1D(d_model, dropout, max_len=256, PE_temp=10000)
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
        tgt = self.word_PE(tgt) # adds positional encoding
        tgt = self.transformer_decoder(tgt, features, tgt_mask=tgt_mask.to(torch.float32), tgt_key_padding_mask=padding_mask.to(torch.float32)) # type match
        output = self.fc_out(tgt)
        return output
    
    def forward(self, src, tgt, tgt_mask):
        features = self.encoder(src)
        output = self.decoder(features, tgt, tgt_mask)
        return output
    
    def greedy_search(self, src, tokenizer, max_seq_len: int=256):
        with torch.no_grad():
            batch_size = src.size(0)
            features = self.encoder(src).detach()
            #print(f"features: {features}")
            tgt = torch.ones(batch_size, 1).long().to(src.device)
            tgt_mask = torch.triu(torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1).to(src.device)

            for i in range(1, max_seq_len):
                output = self.decoder(features, tgt, tgt_mask[:i, :i])
                next_probs = output[:, -1].log_softmax(dim=-1)
                #print(f"next_probs: {next_probs}")
                next_chars = next_probs.argmax(dim=-1, keepdim=True)
                #print(f"next_chars: {next_chars}")
                tgt = torch.cat((tgt, next_chars), dim=1)
            
        return [tokenizer.decode(seq.tolist()) for seq in tgt]
    
    