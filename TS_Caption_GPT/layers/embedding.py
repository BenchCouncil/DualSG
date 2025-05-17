import torch.nn as nn
import torch
import math


class TimeEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)  # 标量 -> d_model维
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        return self.layer_norm(self.linear(x))

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    