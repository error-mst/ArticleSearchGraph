import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # Edge_inddex: [2, num_edges]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, method: str = "dot", hidden_channels: int | None = None):
        super().__init__()
        self.method = method
        if method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels * 3, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )

    def forward(self, z, edge_label_index):
        src, dst = edge_label_index
        zi, zj = z[src], z[dst]

        if self.method == "dot":
            return (zi * zj).sum(dim=-1)
        elif self.method == "mlp":
            hidden_state = torch.cat([zi, zj, zi * zj], dim=-1)
            return self.mlp(hidden_state).squeeze(-1)

class GraphSAGELinkPrediction(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels: int = 64,
                  num_layers: int = 2, decoder: str = "dot"):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, 
                                        out_channels, num_layers=num_layers)
        self.decoder = LinkPredictor(method=decoder, hidden_channels=hidden_channels)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        return self.decoder(z, edge_label_index)