import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import time
import numpy as np
from sklearn.metrics import average_precision_score

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

def train_and_evaluate_node_model(model_class, num_layers, in_channels, out_channels, data, num_epochs=200, lr=1e-2, weight_decay=5e-4):
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

    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        train_accuracy = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

    test_accuracy = test_node_model(data, model)
    print(test_accuracy, training_time)
    return test_accuracy, training_time, train_accuracy

def train_and_evaluate_graph_model(
    model_class, num_layers, in_channels, out_channels, dataset,
    num_epochs=200, lr=1e-2, weight_decay=5e-4, batch_size=64, device=None
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

    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
    train_accuracy = train_correct / train_total
    
    test_loss, test_accuracy = evaluate_graph_model(test_loader, model, criterion, device)
    
    return test_accuracy, training_time, train_accuracy




def train_and_evaluate_lrgb_model(
    model_class, num_layers, in_channels, out_channels, dataset,
    num_epochs=200, lr=1e-3, weight_decay=1e-4, batch_size=32, device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split indices for train/val/test
    num_graphs = len(dataset)
    indices = torch.randperm(num_graphs)
    
    train_idx = indices[:int(0.8 * num_graphs)]
    val_idx = indices[int(0.8 * num_graphs):int(0.9 * num_graphs)]
    test_idx = indices[int(0.9 * num_graphs):]
    
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Calculate positive class weights for loss function
    pos_counts = 0
    total = 0
    for data in train_loader:
        pos_counts += data.y.sum(dim=0)
        total += data.y.size(0)
    pos_weights = ((total - pos_counts) / pos_counts).to(device)
    
    model = model_class(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    def evaluate(loader):
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                # Store raw predictions (before sigmoid) for BCE loss
                y_pred.append(out.cpu().numpy())
                y_true.append(batch.y.cpu().numpy())
        
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        
        # Calculate average precision for each class
        ap_scores = []
        for i in range(y_true.shape[1]):
            # Only calculate AP if there are positive samples
            if y_true[:, i].sum() > 0:
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                ap_scores.append(ap)
        
        # Return mean AP (unweighted average across classes)
        return np.mean(ap_scores)
    
    best_val_score = 0
    best_model = None
    patience = 20
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        val_score = evaluate(val_loader)
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    end_time = time.time()
    training_time = (end_time - start_time) / (epoch + 1)
    
    # Load best model and evaluate
    model.load_state_dict(best_model)
    train_map = evaluate(train_loader)
    test_map = evaluate(test_loader)
    
    print(f"\nFinal Results:")
    print(f"Train mAP: {train_map:.4f}")
    print(f"Test mAP: {test_map:.4f}")
    
    return test_map, training_time, train_map