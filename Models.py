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
        device = 'cuda'
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT) 
        
        self.encoder = nn.Sequential(
            nn.Sequential(*list(densenet.children())[:-1]), # remove the final layer, output (B, 1024, 12, 16)
            nn.Conv2d(1024, d_model, kernel_size=1), # 1x1 convolution, output of (B, d_model, W, H) ex. (1, 256, 12, 16)
            Permute(0, 3, 2, 1), 
            PosEncode2D(d_model=d_model, dropout_percent=dropout, max_len=150, PE_temp=10000), # output (1, 16, 12, 256)
            nn.Flatten(1, 2) 
        ).to(device)
    
        self.tgt_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0) # simple lookup table, input indices 
        self.word_PE = PosEncode1D(d_model, dropout, max_len=150, PE_temp=10000)
        self.transformer_decoder = nn.TransformerDecoder( 
            nn.TransformerDecoderLayer(d_model, nhead, dim_FF, dropout, batch_first=True), # batch_first -> (batch, sequence, feature)
            num_layers,
        ).to(device) # input target and memory (last sequence of the encoder), then tgt_mask, memory_mask
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
        output = self.fc_out(tgt) # size (B, seq_len, vocab_size)
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
    

