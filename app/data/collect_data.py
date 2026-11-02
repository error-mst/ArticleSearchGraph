from torch_geometric.datasets import Planetoid

def get_dataset():
    return Planetoid(root="/cora", name="Cora", split="full")
