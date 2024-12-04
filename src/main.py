from models import GCNModel, GATModel, GINModel, GCNGraphClassifier, GATGraphClassifier, GINGraphClassifier
from train import train_and_evaluate_node_model, train_and_evaluate_graph_model
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
    parser.add_argument('--models', nargs='+', default=['GCN', 'GAT', 'GIN'],
                        choices=['GCN', 'GAT', 'GIN', 'all'],
                        help="Specify which models to run: GCN, GAT, GIN, or all")
    args = parser.parse_args()

    # Handle 'all' argument
    if 'all' in args.models:
        selected_models = ['GCN', 'GAT', 'GIN']
    else:
        selected_models = args.models

    # Datasets
    datasets = {
        "Cora": load_dataset(root="/tmp/Cora", name="Cora", device=device),
        "IMDB-BINARY": load_dataset(root='/tmp/IMDB', name='IMDB-BINARY', device=device),
        "ENZYMES": load_dataset(root='/tmp/ENZYMES', name='ENZYMES', device=device)
    }

    layer_configs = [2, 3]
    model_classes = {
        "Cora": {"GCN": GCNModel, "GAT": GATModel, "GIN": GINModel},
        "IMDB-BINARY": {"GCN": GCNGraphClassifier, "GAT": GATGraphClassifier, "GIN": GINGraphClassifier},
        "ENZYMES": {"GCN": GCNGraphClassifier, "GAT": GATGraphClassifier, "GIN": GINGraphClassifier}
    }

    # Filter models based on selected_models
    for dataset_name in model_classes:
        model_classes[dataset_name] = {
            k: v for k, v in model_classes[dataset_name].items() if k in selected_models
        }

    results = {
        dataset_name: {
            model_name: {'accuracy': [], 'time': []}
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
        else:
            data = dataset  # Data is already on the correct device from load_dataset
            num_classes = dataset.num_classes
            num_features = dataset.num_features

        for model_name, model_class in model_classes[dataset_name].items():
            print(f"\nTraining {model_name} on {dataset_name}")
            for num_layers in layer_configs:
                if dataset_name == "Cora":
                    accuracy, training_time = train_and_evaluate_node_model(
                        model_class, num_layers, num_features, num_classes, data
                    )
                else:
                    accuracy, training_time = train_and_evaluate_graph_model(
                        model_class, num_layers, num_features, num_classes, data,
                        device=device
                    )
                results[dataset_name][model_name]['accuracy'].append(accuracy)
                results[dataset_name][model_name]['time'].append(training_time)

    # Write results to a log file
    log_file_path = "../outputs/results_log.txt"
    with open(log_file_path, 'w') as log_file:
        for dataset_name, metrics in results.items():
            log_file.write(f"Results for {dataset_name} dataset:\n")
            for model_name, values in metrics.items():
                log_file.write(f"  Model: {model_name}\n")
                log_file.write(f"    Accuracies: {values['accuracy']}\n")
                log_file.write(f"    Training Times: {values['time']}\n")
            log_file.write("\n")

    # Set style for better-looking plots
    plt.style.use('seaborn')
    
    # Create plots
    fig, axes = plt.subplots(len(datasets), 2, figsize=(15, 6 * len(datasets)))
    
    # Use a different color for each model
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (dataset_name, metrics) in enumerate(results.items()):
        # Plot accuracies
        for j, (model_name, values) in enumerate(metrics.items()):
            axes[i, 0].plot(layer_configs, values['accuracy'], 
                          label=f'{model_name}', 
                          marker='o', 
                          color=colors[j],
                          linewidth=2,
                          markersize=8)
        axes[i, 0].set_title(f'{dataset_name} - Accuracy vs Number of Layers', 
                           fontsize=12, pad=15)
        axes[i, 0].set_xlabel('Number of Layers', fontsize=10)
        axes[i, 0].set_ylabel('Accuracy', fontsize=10)
        axes[i, 0].grid(True, linestyle='--', alpha=0.7)
        axes[i, 0].legend(fontsize=10)
        axes[i, 0].set_ylim(0, 1)  # Accuracy is between 0 and 1

        # Plot training times
        for j, (model_name, values) in enumerate(metrics.items()):
            axes[i, 1].plot(layer_configs, values['time'], 
                          label=f'{model_name}', 
                          marker='o', 
                          color=colors[j],
                          linewidth=2,
                          markersize=8)
        axes[i, 1].set_title(f'{dataset_name} - Training Time vs Number of Layers', 
                           fontsize=12, pad=15)
        axes[i, 1].set_xlabel('Number of Layers', fontsize=10)
        axes[i, 1].set_ylabel('Training Time (seconds)', fontsize=10)
        axes[i, 1].grid(True, linestyle='--', alpha=0.7)
        axes[i, 1].legend(fontsize=10)

    plt.tight_layout()
    
    # Save the plot
    plot_path = "../outputs/model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    main()