import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool,  global_mean_pool, global_max_pool
from torch_geometric.nn import Linear
from torch_geometric.nn import TransformerConv
import torch.nn as nn


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


class GPSLayer(torch.nn.Module):
    """
    Graph Positional and Structural Layer that combines local GNN with transformer for global attention
    """
    def __init__(self, dim_h, heads=4):
        super().__init__()
        # Local MPNN
        self.local_model = GCNConv(dim_h, dim_h, add_self_loops=False)
        
        # Global attention - using TransformerConv for attention mechanism
        self.global_model = TransformerConv(
            dim_h, 
            dim_h // heads, 
            heads=heads, 
            concat=True,
            beta=True  # Enable edge attention
        )
        
        # Layer norm for stability
        self.norm1 = nn.LayerNorm(dim_h)
        self.norm2 = nn.LayerNorm(dim_h)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim_h, 2 * dim_h),
            nn.ReLU(),
            nn.Linear(2 * dim_h, dim_h)
        )
        
        # Parameters for combining local and global
        self.local_weight = nn.Parameter(torch.ones(1))
        self.global_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x, edge_index):
        # Ensure proper data types
        x = x.float()
        edge_index = edge_index.long()
        
        # Local MPNN
        local_out = self.local_model(x, edge_index)
        
        # Global attention
        global_out = self.global_model(x, edge_index)
        
        # Combine local and global (adaptive weights)
        x = self.local_weight * local_out + self.global_weight * global_out
        
        # First normalization and residual
        x = self.norm1(x + x)
        
        # FFN and second normalization
        out = self.ffn(x)
        out = self.norm2(out + x)
        
        return out

class GPSNode(torch.nn.Module):
    """GPS model for node classification"""
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        # Initial projection
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # GPS layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GPSLayer(hidden_channels))
        
        # Output projection
        self.output_proj = Linear(hidden_channels, out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Initial projection
        x = self.input_proj(x.float())
        
        # GPS layers
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return F.log_softmax(x, dim=1)

class GPSGraph(torch.nn.Module):
    """GPS model for graph classification"""
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        # Initial projection
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # GPS layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GPSLayer(hidden_channels))
        
        # Multiple pooling for better graph representation
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool
        
        # Output projections
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial projection
        x = self.input_proj(x.float())
        
        # GPS layers
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
        
        # Multiple pooling operations
        pool_mean = self.pool_mean(x, batch)
        pool_max = self.pool_max(x, batch)
        
        # Concatenate different pooling results
        x = torch.cat([pool_mean, pool_max], dim=1)
        
        # Output projection
        x = self.output_proj(x)
        
        return F.log_softmax(x, dim=1)