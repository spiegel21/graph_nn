from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import Planetoid, TUDataset

def find_max_degree(dataset):
    max_degree = 0
    for data in dataset:
        degree = data.edge_index[0].bincount().max().item()
        if degree > max_degree:
            max_degree = degree
    return max_degree

# Load dataset with OneHotDegree transform using maximum degree
def load_dataset(root, name):
    if name == 'Cora':
        tmp_dataset = Planetoid(root=root, name=name)
        max_degree = find_max_degree(tmp_dataset)
        return Planetoid(root=root, name=name, transform=OneHotDegree(max_degree))
    else:
        tmp_dataset = TUDataset(root=root, name=name)
        max_degree = find_max_degree(tmp_dataset)
        return TUDataset(root=root, name=name, transform=OneHotDegree(max_degree))