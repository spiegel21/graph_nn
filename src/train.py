import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import time

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

def train_graph_model(loader, model, optimizer, criterion, device):
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

def evaluate_graph_model(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return total_loss / len(loader), correct / total

def train_and_evaluate_node_model(model_class, num_layers, in_channels, out_channels, data, num_epochs=200, lr=1e-3, weight_decay=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(in_channels, out_channels=out_channels, num_layers=num_layers).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    start_time = time.time()
    
    best_val_acc = 0
    best_model = None
    patience_counter = 0
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_node_model(data, model, optimizer, criterion)
        val_loss, val_accuracy = evaluate_node_model(data, model, criterion)
        # print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        if(val_accuracy > best_val_acc):
            best_val_acc = val_accuracy
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if(patience_counter >= 20):
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model)
            break

    end_time = time.time()
    training_time = (end_time - start_time) / (epoch + 1)
    test_accuracy = test_node_model(data, model)
    print(test_accuracy, training_time)
    return test_accuracy, training_time

def train_and_evaluate_graph_model(
    model_class, num_layers, in_channels, out_channels, dataset,
    num_epochs=200, lr=1e-3, weight_decay=5e-4, batch_size=64, device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    
    # Split the dataset
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model and move to device
    model = model_class(in_channels, out_channels=out_channels, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    
    start_time = time.time()
    
    best_val_acc = 0
    best_model = None
    patience_counter = 0
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_graph_model(train_loader, model, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate_graph_model(val_loader, model, criterion, device)
        if(val_accuracy > best_val_acc):
            best_val_acc = val_accuracy
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if(patience_counter >= 20):
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model)
            break
        # print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    end_time = time.time()
    training_time = (end_time - start_time) / (epoch + 1)
    
    test_loss, test_accuracy = evaluate_graph_model(test_loader, model, criterion, device)
    
    return test_accuracy, training_time