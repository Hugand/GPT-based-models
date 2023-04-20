import torch
from torch import nn
from modules.CausalSelfAttention import CausalSelfAttention
from modules.FullyConnectedLayers import FullyConnectedLayers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, embedding_dim, bias=False, dropout=0.1, block_size=1024):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // n_heads
        self.attention_projection = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        # self.multihead_attention = CausalSelfAttention(embedding_dim, n_heads)
        self.multihead_attention = CausalSelfAttention(embedding_dim, n_heads, dropout, bias, block_size)
        #  n_embd, n_head, dropout, bias, block_size
        self.layernorm_1 = nn.LayerNorm(embedding_dim).to(device)
        self.fc_layers = FullyConnectedLayers(embedding_dim, dropout).to(device)
        self.layernorm_2 = nn.LayerNorm(embedding_dim).to(device)

    def forward(self, x):
        residual = x
        x = self.multihead_attention(x)
        x = self.layernorm_1(x + residual)
        residual  = x
        x = self.fc_layers(x)
        x = self.layernorm_2(x + residual)
        return x