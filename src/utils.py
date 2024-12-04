import torch
from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import Compose

def find_max_degree(dataset):
    max_degree = 0
    for data in dataset:
        degree = data.edge_index[0].bincount().max().item()
        if degree > max_degree:
            max_degree = degree
    return max_degree

class ToDevice(object):
    def __init__(self, device):
        self.device = device
    
    def __call__(self, data):
        return data.to(self.device)

# Load dataset with OneHotDegree transform using maximum degree
def load_dataset(root, name, device):
    if name == 'Cora':
        # First load without transform to find max degree
        tmp_dataset = Planetoid(root=root, name=name)
        max_degree = find_max_degree(tmp_dataset)
        
        # Create transform pipeline
        transform = Compose([
            OneHotDegree(max_degree),
            ToDevice(device)
        ])
        
        # Load dataset with transforms
        return Planetoid(root=root, name=name, transform=transform)
    else:
        # First load without transform to find max degree
        tmp_dataset = TUDataset(root=root, name=name)
        max_degree = find_max_degree(tmp_dataset)
        
        # Create transform pipeline
        transform = Compose([
            OneHotDegree(max_degree),
            ToDevice(device)
        ])
        
        # Load dataset with transforms
        return TUDataset(root=root, name=name, transform=transform)