import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures
from torch_geometric.utils import negative_sampling

from link_predictor import GraphSAGELinkPrediction
from sklearn.metrics import roc_auc_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Planetoid(root="data/Cora", name="Cora", transform=NormalizeFeatures())
data = dataset[0]

transform = RandomLinkSplit(
    num_val = 0.1,
    num_test = 0.2,
    is_undirected = True,
    add_negative_train_samples=False
)

train_data, val_data, test_data = transform(data)

model = GraphSAGELinkPrediction(
    in_channels=dataset.num_node_features,
    hidden_channels=128,
    out_channels=128,
    num_layers=3,
    method="mlp"
).to(device)

optimzer = torch.optim.Adam(model.parameters(), lr=0.01)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

def get_link_labels(pos_edge_index, neg_edge_index):
    pos_sz = pos_edge_index.size(1)
    neg_sz = neg_edge_index.size(1)
    labels = torch.zeros(pos_sz + neg_sz, device=pos_edge_index.device)
    labels[:pos_sz] = 1.0
    return labels

def train():
    model.train()
    optimzer.zero_grad()

    z = model.encode(train_data.x, train_data.edge_index)
    pos_edge_index = train_data.edge_label_index

    neg_edge_index = negative_sampling(
        edge_index = train_data.edge_index,
        num_nodes = train_data.num_nodes,
        num_neg_samples = pos_edge_index.size(1),
        method="sparse",
    )

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_label = get_link_labels(pos_edge_index, neg_edge_index)

    logits = model.decode(z, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, edge_label)
    loss.backward()
    optimzer.step()
    return float(loss)

@torch.no_grad()
def evaluate(split_data):
    model.eval()
    z = model.encode(split_data.x, split_data.edge_index)
    logits = model.decode(z, split_data.edge_label_index)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = split_data.edge_label.cpu().numpy()

    return roc_auc_score(labels, probs)

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        val_auc = evaluate(val_data)
        test_auc = evaluate(test_data)
        print(f"Epoch {epoch:3d} | loss {loss:.4f} | val AUC {val_auc:.4f} | test AUC {test_auc:.4f}")