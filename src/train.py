import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_node_model(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_node_model(data, model, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
        pred = out.argmax(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
        total = data.val_mask.sum().item()
        accuracy = correct / total
    return loss, accuracy

def test_node_model(data, model):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        total = data.test_mask.sum().item()
        accuracy = correct / total
    return accuracy

def train_graph_model(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_graph_model(loader, model, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            # Move the entire batch to the specified device
            batch = batch.to(device)

            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
    total_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return total_loss, accuracy

def train_and_evaluate_node_model(model_class, num_layers, in_channels, out_channels, data, num_epochs=200, lr=1e-2, weight_decay=5e-4):
    model = model_class(in_channels, out_channels=out_channels, num_layers=num_layers).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_node_model(data, model, optimizer, criterion)
        val_loss, val_accuracy = evaluate_node_model(data, model, criterion)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    end_time = time.time()
    training_time = end_time - start_time
    test_accuracy = test_node_model(data, model)
    return test_accuracy, training_time

def train_and_evaluate_graph_model(
    model_class, num_layers, in_channels, out_channels, dataset,
    num_epochs=200, lr=1e-2, weight_decay=5e-4, batch_size=64, device='cpu'
):
    torch.manual_seed(42)
    dataset = dataset.shuffle()  # Shuffle the dataset
    
    # Split the dataset into train, val, and test subsets
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]
    
    # Create data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize the model and move it to the device
    model = model_class(in_channels, out_channels=out_channels, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Train the model
        train_loss = train_graph_model(train_loader, model, optimizer, criterion)
        
        # Evaluate on the validation set
        val_loss, val_accuracy = evaluate_graph_model(val_loader, model, criterion)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Test the model
    test_loss, test_accuracy = evaluate_graph_model(test_loader, model, criterion)
    
    return test_accuracy, training_time
