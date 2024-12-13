import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool
from torch_geometric.nn import Linear
from torch_geometric.nn import TransformerConv

class GINModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super(GINModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        # Simplified MLP in GINConv - single layer instead of two
        for i in range(num_layers):
            if i == 0:
                nn = torch.nn.Linear(in_channels, hidden_channels)
            elif i == num_layers - 1:
                nn = torch.nn.Linear(hidden_channels, out_channels)
            else:
                nn = torch.nn.Linear(hidden_channels, hidden_channels)
            # Disable eps training to remove adaptivity
            conv = GINConv(nn, train_eps=False, eps=0.0)
            self.layers.append(conv)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super(GCNModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        # Remove improved parameter
        if num_layers > 1:
            self.layers.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
            self.layers.append(GCNConv(hidden_channels, out_channels, add_self_loops=False))
        else:
            self.layers.append(GCNConv(in_channels, out_channels, add_self_loops=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, heads=1, num_layers=2):
        super(GATModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        # Reduce heads to 1 and remove concatenation
        if num_layers > 1:
            self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False))
            self.layers.append(GATConv(hidden_channels, out_channels, heads=heads, concat=False))
        else:
            self.layers.append(GATConv(in_channels, out_channels, heads=heads, concat=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GINGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super(GINGraphClassifier, self).__init__()
        self.layers = torch.nn.ModuleList()
        # Simplified MLP in GINConv - single layer instead of two
        for i in range(num_layers):
            if i == 0:
                nn = torch.nn.Linear(in_channels, hidden_channels)
            else:
                nn = torch.nn.Linear(hidden_channels, hidden_channels)
            conv = GINConv(nn, train_eps=False, eps=0.0)
            self.layers.append(conv)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        device = next(self.parameters()).device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=None, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers > 1:
            self.layers.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
            self.layers.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
        else:
            self.layers.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        device = next(self.parameters()).device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class GATGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=None, heads=1, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers > 1:
            self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False))
            self.layers.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False))
        else:
            self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        device = next(self.parameters()).device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GPSNode(torch.nn.Module):
    """Simplified GPS model specifically for node classification"""
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.out = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
        # Output layer
        x = self.out(x, edge_index)
        return F.log_softmax(x, dim=1)

    def forward(self, data):
        # Ensure we keep node-level predictions
        x = data.x  # Shape: [num_nodes, num_features]
        edge_index = data.edge_index
        
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
        # Output layer - should maintain [num_nodes, num_classes] shape
        x = self.out(x, edge_index)
        return F.log_softmax(x, dim=1)

class GPSGraph(torch.nn.Module):
    """GPS model for graph classification"""
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
