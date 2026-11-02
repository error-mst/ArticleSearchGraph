import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing

class SAGEConv(MessagePassing):
    def __init__(self, in_features, out_features, aggr: str = "mean",
                 dropout: float = 0.3, activation: bool = False):
        super().__init__(aggr=None)
        self.linear = nn.Linear(in_features, out_features)

    def _aggr(self, inputs: Tensor, index: Tensor):
