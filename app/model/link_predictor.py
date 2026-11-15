import torch
import torch.nn as nn
from torch_model import GraphSAGEEncoder
from my_model import GraphSAGE

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
                  num_layers: int = 2, impl: str = "my_own", method: str = "dot"):
        super().__init__()

        if impl == "torch":
            self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, 
                                            out_channels, num_layers=num_layers)
            self.decoder = LinkPredictor(method=method, hidden_channels=hidden_channels)
        elif impl == "my_own":
            self.encoder = GraphSAGE(in_channels, hidden_channels, out_channels, 
                                     num_layers=num_layers)
            self.decoder = LinkPredictor(method=method, hidden_channels=hidden_channels)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        return self.decoder(z, edge_label_index)