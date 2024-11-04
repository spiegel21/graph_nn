import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool

class GINModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2, dropout=0.5):
        super(GINModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU()
                )
            elif i == num_layers - 1:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, out_channels)
                )
            else:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU()
                )
            conv = GINConv(nn, train_eps=True)
            self.layers.append(conv)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2, dropout=0.5):
        super(GCNModel, self).__init__()
        self.gcn_layers = torch.nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_layers.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, heads=8, num_layers=2, dropout=0.5):
        super(GATModel, self).__init__()
        self.gat_layers = torch.nn.ModuleList()
        self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.gat_layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

# GIN Model for Graph Classification
class GINGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2, dropout=0.5):
        super(GINGraphClassifier, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU()
                )
            else:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU()
                )
            conv = GINConv(nn, train_eps=True)
            self.layers.append(conv)
        self.dropout = dropout
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)  # Aggregate features to graph level
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# GCN Model for Graph Classification
class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=None, num_layers=2, dropout=0.5):
        super().__init__()
        self.gcn_layers = torch.nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)  # Aggregate features to graph level
        return F.log_softmax(self.fc(x), dim=1)

# GAT Model for Graph Classification
class GATGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=None, heads=8, num_layers=2, dropout=0.5):
        super().__init__()
        self.gat_layers = torch.nn.ModuleList()
        self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)  # Aggregate features to graph level
        return F.log_softmax(self.fc(x), dim=1)