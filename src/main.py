from models import GCNModel, GATModel, GINModel, GCNGraphClassifier, GATGraphClassifier, GINGraphClassifier
from train import train_and_evaluate_node_model, train_and_evaluate_graph_model
from utils import load_dataset
import matplotlib.pyplot as plt
import argparse
import torch

def main():
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
        "Cora": load_dataset(root="/tmp/Cora", name="Cora"),
        "IMDB-BINARY": load_dataset(root='/tmp/IMDB', name='IMDB-BINARY'),
        "ENZYMES": load_dataset(root='/tmp/ENZYMES', name='ENZYMES')
    }

    layer_configs = [2, 3, 4]
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
            data = dataset[0].to(device)  # Move data to device
            num_classes = dataset.num_classes
            num_features = data.num_node_features
        else:
            data = [graph.to(device) for graph in dataset]
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
                        model_class, num_layers, num_features, num_classes, data
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

    # Plotting results
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 18))

    for i, (dataset_name, metrics) in enumerate(results.items()):
        # Plot accuracies
        for model_name, values in metrics.items():
            axes[i, 0].plot(layer_configs, values['accuracy'], label=f'{model_name} Accuracy', marker='o')
        axes[i, 0].set_title(f'{dataset_name} - Accuracy vs Number of Layers')
        axes[i, 0].set_xlabel('Number of Layers')
        axes[i, 0].set_ylabel('Accuracy')
        axes[i, 0].legend()

        # Plot training times
        for model_name, values in metrics.items():
            axes[i, 1].plot(layer_configs, values['time'], label=f'{model_name} Training Time', marker='o')
        axes[i, 1].set_title(f'{dataset_name} - Training Time vs Number of Layers')
        axes[i, 1].set_xlabel('Number of Layers')
        axes[i, 1].set_ylabel('Training Time (seconds)')
        axes[i, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
