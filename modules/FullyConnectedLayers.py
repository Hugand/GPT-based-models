import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FullyConnectedLayers(nn.Module):
    def __init__(self, base_size, dropout, bias=True):
        super().__init__()
        self.fc_input = nn.Sequential(
            nn.Linear(base_size, 4 * base_size, bias=bias),
            nn.GELU()
        )
        self.fc_output = nn.Sequential(
            nn.Linear(4 * base_size, base_size, bias=bias),
            nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_input(x)
        x = self.fc_output(x)
        x = self.dropout(x)
        return x
    