import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence
from torch import Tensor

# Класс для слоя GraphSAGE
# Количество слоев затем будет принятым за k
class GraphSAGELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 aggregator: str = "pool", dropout: float=0.1,
                 activation: bool = True):
        super().__init_()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator = aggregator
        self.dropout = dropout
        self.activation = activation

        if aggregator == "pool":
            self.pool = nn.Linear(in_channels, out_channels)
        elif aggregator == "lstm":
            self.lstm = nn.LSTM(input_size=in_channels, hidden_size=in_channels, batch_first=True)

        self.update_linear = nn.Linear(2 * in_channels, out_channels)


    def _aggregate(self, neighbor_features: Tensor, mask: Tensor) -> Tensor:
        num_nodes, _, dim = neighbor_features.shape

        if self.aggregator == "mean":
            mask_ = mask.unsqueeze(-1).to(neighbor_features.dtype)
            total_nodes = mask_.sum(dim=1).clamp_min(1.0)
            sum_emb = (neighbor_features * mask).sum(dim=1)
            aggregated = sum_emb / total_nodes

        elif self.aggregator == "pool":
            pool = self.pool(neighbor_features).relu()
            min_v = torch.finfo(pool.dtype).min
            pool = pool.masked_fill(~mask.unsqueeze(-1), min_v)

            pool = pool.max(dim=1)
            aggregated = torch.where(
                mask.any(dim=1).unsqueeze(-1),
                pool,
                torch.zeros_like(pool),
            )

        elif self.aggregator == "lstm":
            aggregated = neighbor_features.new_zeros((num_nodes, dim))
            # Получаем узлы для вычисления
            nodes_with_nodes = mask.any(dim=1).nonzero(as_tuple=False).view(-1)

            if nodes_with_nodes.numel() > 0:
                seq = neighbor_features[nodes_with_nodes]
                lens = mask[nodes_with_nodes].sum(dim=1)
                packed_seq = nn.utils.rnn.pack_padded_sequence(
                    seq,
                    lengths=lens,
                    batch_first=True,
                    enforce_sorted=False
                )
                _, (hidden, _) = self.lstm(packed_seq)
                aggregated[nodes_with_nodes] = hidden[-1]

        return aggregated

    def forward(self, x: Tensor, neighbor_index: Tensor, neighbor_mask: Tensor) -> Tensor:

        neighbor_features = x[neighbor_index]
        neighbor_features =F.dropout(neighbor_features, p=self.dropout, training=self.training)
        self_features = F.dropout(x, p=self.dropout, training=self.training)

        aggregated = self._aggregate(neighbor_features, neighbor_mask)
        combined = torch.cat([self_features, aggregated], dim=-1)

        output = self.update_linear(combined)
        if self.activation:
            output = output.relu()

        return output

class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, num_layers: int = 2,
                 *, aggregator: str = "mean", neighbor_cnt: Sequence[int] | None = None,
                 dropout: float = 0.2, normalize: bool = True):
        super().__init__()

        if neighbor_cnt is None:
            neighbor_cnt = (10 for _ in range(num_layers))

        dims = [in_channels]
        if num_layers > 1:
            dims.extend([hidden_channels] * (num_layers - 1))
        dims.append(out_channels)

        self.layers = nn.ModuleList([GraphSAGELayer(dims[i], dims[i + 1],
                                                    aggregator=aggregator,
                                                    dropout=dropout)
                                  for i in range(num_layers)])
        self.neighbor = tuple(neighbor_cnt)
        self.normalize = normalize

    def forward(self, x: Tensor, sampled: Sequence[tuple[Tensor, Tensor]]):
        if len(sampled) != len(self.layers):
            raise ValueError("Sampled должен содержать сэмплы для каждого прыжка (Слоя аггрегации)")

        hidden_state = x
        for layer, (neighbor_index, neighbor_mask) in zip(self.layers, sampled):
            neighbor_index = neighbor_index.to(hidden_state.device)
            neighbor_mask = neighbor_mask.to(hidden_state.device)

            hidden_state = layer(hidden_state, neighbor_index, neighbor_mask)
            if self.normalize:
                hidden_state = F.normalize(hidden_state, p=2, dim=-1)

        return hidden_state


class LinkPredictor(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 128, *,
                 directed: bool = False, dropout: float = 0.2):
        super().__init__()
        self.directed = directed
        feature_dim = 4 * emb_dim if directed else 2 * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, edge_idx: Tensor):
        src, dst = edge_idx
        z_src, z_dst = z[src], z[dst]
        features = [torch.abs(z_src - z_dst), z_src * z_dst]
        if self.directed:
            features = features = [z_src, z_dst, *features]
        return self.mlp(torch.cat(features, dim=-1)).squeeze(-1)
