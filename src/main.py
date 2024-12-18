from models import *
from train import train_and_evaluate_node_model, train_and_evaluate_graph_model, train_and_evaluate_lrgb_model
from utils import load_dataset
import matplotlib.pyplot as plt
import argparse
import torch
import os

def main():
    # Create outputs directory if it doesn't exist
    os.makedirs("../outputs", exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['GCN', 'GAT', 'GIN', 'GPS'],
                        choices=['GCN', 'GAT', 'GIN', 'GPS', 'all'],
                        help="Specify which models to run: GCN, GAT, GIN, 'GPS' or all")
    args = parser.parse_args()

    # Handle 'all' argument
    if 'all' in args.models:
        selected_models = ['GCN', 'GAT', 'GIN', 'GPS']
    else:
        selected_models = args.models

    # Datasets
    datasets = {
        "Cora": load_dataset(root="/tmp/Cora", name="Cora", device=device),
        "IMDB-BINARY": load_dataset(root='/tmp/IMDB', name='IMDB-BINARY', device=device),
        "ENZYMES": load_dataset(root='/tmp/ENZYMES', name='ENZYMES', device=device),
        "LRGB": load_dataset(root='/tmp/LRGB', name='LRGB', device=device)
    }

    layer_configs = list(range(2, 22, 2))

    model_classes = {
        "Cora": {"GCN": GCNModel, "GAT": GATModel, "GIN": GINModel, "GPS": GPSNode},
        "IMDB-BINARY": {"GCN": GCNGraphClassifier, "GAT": GATGraphClassifier, 
                       "GIN": GINGraphClassifier, "GPS": GPSGraph},
        "ENZYMES": {"GCN": GCNGraphClassifier, "GAT": GATGraphClassifier, 
                   "GIN": GINGraphClassifier, "GPS": GPSGraph},
        "LRGB": {"GCN": GCNGraphClassifier, "GAT": GATGraphClassifier, 
                 "GIN": GINGraphClassifier, "GPS": GPSGraph}
    }

    # Filter models based on selected_models
    for dataset_name in model_classes:
        model_classes[dataset_name] = {
            k: v for k, v in model_classes[dataset_name].items() if k in selected_models
        }

    results = {
        dataset_name: {
            model_name: {'accuracy': [], 'time': [], 'train_acc': []}
            for model_name in model_classes[dataset_name]
        }
        for dataset_name in datasets
    }

    for dataset_name, dataset in datasets.items():
        print(f"\nEvaluating on {dataset_name} dataset")
        
        
        if dataset_name == "Cora":
            data = dataset[0]  # Data is already on the correct device from load_dataset
            num_classes = dataset.num_classes
            num_features = data.num_node_features
        elif dataset_name == "LRGB":
            data = dataset
            num_classes = dataset.num_classes  # Now using the attribute we added
            num_features = dataset[0].num_features
            print(f"LRGB dataset details: {num_features} features, {num_classes} targets")
        else:
            data = dataset  # Data is already on the correct device from load_dataset
            num_classes = dataset.num_classes
            num_features = dataset.num_features

        for model_name, model_class in model_classes[dataset_name].items():
            print(f"\nTraining {model_name} on {dataset_name}")

            

            for num_layers in layer_configs:

                if dataset_name == "LRGB" and num_layers > 3:
                    break

                if dataset_name == "Cora":  
                    accuracy, training_time, train_acc = train_and_evaluate_node_model(
                        model_class, num_layers, num_features, num_classes, data
                    )
                elif dataset_name == "LRGB":
                    accuracy, training_time, train_acc = train_and_evaluate_lrgb_model(
                        model_class, num_layers, num_features, num_classes, data,
                        device=device
                    )
                else:
                    accuracy, training_time, train_acc = train_and_evaluate_graph_model(
                        model_class, num_layers, num_features, num_classes, data,
                        device=device
                    )
                
                # Store results
                results[dataset_name][model_name]['train_acc'].append(train_acc)
                results[dataset_name][model_name]['accuracy'].append(accuracy)
                results[dataset_name][model_name]['time'].append(training_time)



    # Write results to a log file
    log_file_path = "../outputs/results_log.txt"
    with open(log_file_path, 'w') as log_file:
        for dataset_name, metrics in results.items():
            log_file.write(f"Results for {dataset_name} dataset:\n")
            for model_name, values in metrics.items():
                log_file.write(f"  Model: {model_name}\n")
                log_file.write(f"  Train Accuracies: {values['train_acc']}\n")
                log_file.write(f"    Accuracies: {values['accuracy']}\n")
                log_file.write(f"    Training Times: {values['time']}\n")
            log_file.write("\n")

    # Create plots
    fig, axes = plt.subplots(len(datasets), 2, figsize=(15, 6 * len(datasets)))
    
    filtered_results = {k: v for k, v in results.items() if k != 'LRGB'} 

    # Use a different color for each model
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
 

    for i,  (dataset_name, metrics) in enumerate(filtered_results.items()):
        # Plot accuracies
        ax1 = axes[i, 0]
        for j, (model_name, values) in enumerate(metrics.items()):
            ax1.plot(layer_configs, values['accuracy'], 
                    label=f'{model_name}', 
                    marker='o', 
                    color=colors[j],
                    linewidth=2,
                    markersize=8)
        ax1.set_title(f'{dataset_name} - Accuracy vs Number of Layers', 
                     fontsize=12, pad=15)
        ax1.set_xlabel('Number of Layers', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, 1)  # Accuracy is between 0 and 1

        # Plot training times
        ax2 = axes[i, 1]
        for j, (model_name, values) in enumerate(metrics.items()):
            ax2.plot(layer_configs, values['time'], 
                    label=f'{model_name}', 
                    marker='o', 
                    color=colors[j],
                    linewidth=2,
                    markersize=8)
        ax2.set_title(f'{dataset_name} - Training Time vs Number of Layers', 
                     fontsize=12, pad=15)
        ax2.set_xlabel('Number of Layers', fontsize=10)
        ax2.set_ylabel('Training Time per Epoch (seconds)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)

    plt.tight_layout()
    
    # Save the plot
    plot_path = "../outputs/model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    main()