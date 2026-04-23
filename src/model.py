import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class Transformer(nn.Module):
    def __init__(self, d_in, embedd_dim, n_heads, d_ff, num_layers, dropout, max_seq_len):
        super().__init__()

        self.emdedding_dimension = nn.Linear(d_in, embedd_dim)

        pe = torch.zeros(max_seq_len, embedd_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedd_dim, 2).float() * (-math.log(10000.0) / embedd_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embedd_dim,
            nhead = n_heads,
            dim_feedforward = d_ff,
            dropout = dropout,
            activation = 'relu',
            batch_first = True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)
        self.reg_head = nn.Linear(embedd_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.emdedding_dimension(x)
        x = self.dropout(x + self.pe[:, :x.size(1), :])
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.reg_head(x)

        return x