import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing

# mean aggregation
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr=None)
        self.linear_neighbor = nn.Linear(in_channels, out_channels)
        self.linear_update = nn.Linear(in_channels + out_channels, out_channels)

    def reset_parameters(self):
        return self.linear_neighbor.reset_parameters()
        return self.linear_update.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor):
        x_transformed = self.linear_neighbor(x)
        return self.propagate(edge_index, x=x, x_transformed=x_transformed)

    def aggregate(self, inputs: Tensor, index: Tensor, dim_size: int | None = None):
        if not dim_size:
            dim_size = index.max().item() + 1
        in_channels = inputs.size(1) # Число входных признаков

        sum = inputs.new_zeros((dim_size, in_channels))
        sum.index_add_(0, index, inputs)

        cnt = torch.zeros(dim_size, 1, dtype=inputs.dtype, device=inputs.device)
        cnt.index_add_(0, index, torch.ones_like(index, dtype=inputs.dtype).unsqueeze(1))
        cnt = cnt.clamp(min=1.0)
        return sum / cnt

    def message(self, x_j: Tensor):
        return x_j

    def update(self, aggr, x: Tensor):
        concatenated = torch.cat([x, aggr], dim=-1)
        return self.linear_update(concatenated)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels: int = 128, out_channels: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # Наполнение слоями
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
        return x