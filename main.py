import os
from models import GCNModel, GATModel, GINModel, GCNGraphClassifier, GATGraphClassifier, GINGraphClassifier
from train import train_and_evaluate_node_model, train_and_evaluate_graph_model
from utils import load_dataset
import matplotlib.pyplot as plt

def main():
    # Datasets
    datasets = {
        "Cora": load_dataset(root="/tmp/Cora", name="Cora"),  # Cora has 7 classes
        "IMDB-BINARY": load_dataset(root='/tmp/IMDB', name='IMDB-BINARY'),  # IMDB-BINARY has 2 classes
        "ENZYMES": load_dataset(root='/tmp/ENZYMES', name='ENZYMES')  # ENZYMES has 6 classes
    }
    
    layer_configs = [2, 4, 8]
    model_classes = {
        "Cora": {"GCN": GCNModel, "GAT": GATModel, "GIN": GINModel},
        "IMDB-BINARY": {"GCN": GCNGraphClassifier, "GAT": GATGraphClassifier, "GIN": GINGraphClassifier},
        "ENZYMES": {"GCN": GCNGraphClassifier, "GAT": GATGraphClassifier, "GIN": GINGraphClassifier}
    }
    
    results = {dataset_name: {model_name: {'accuracy': [], 'time': []} for model_name in model_classes[dataset_name].keys()} for dataset_name in datasets.keys()}
    
    for dataset_name, dataset in datasets.items():
        print(f"\nEvaluating on {dataset_name} dataset")
        
        if dataset_name == "Cora":
            data = dataset[0]
            num_classes = dataset.num_classes
            num_features = data.num_node_features
        else:
            data = dataset
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
    os.exit(0)

if __name__ == "__main__":
    main()
