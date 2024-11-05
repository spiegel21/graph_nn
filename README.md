# Graph Neural Networks (GNNs) - Survey and Experiments

This repository contains code and resources for a capstone project exploring various Graph Neural Network (GNN) models. The project evaluates the performance of GNN architectures, specifically Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and Graph Isomorphism Networks (GINs), on diverse graph-based tasks. The focus is on understanding each model's effectiveness, limitations, and applicability across different datasets.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Graph Neural Networks have emerged as powerful tools for analyzing graph-structured data, with applications in social network analysis, biology, recommendation systems, and more. This project investigates GNNs on tasks like node classification and graph classification, using benchmark datasets (Cora, IMDB-Binary, ENZYMES). The models are evaluated for their performance and behavior, particularly under different layer depths, to identify patterns of oversmoothing, oversquashing, and model expressiveness limitations.

## Datasets

The following datasets are used:
- **Cora**: A citation network dataset for node classification, with nodes representing publications and edges representing citations.
- **IMDB-Binary**: A graph classification dataset with ego networks of actors, where each graph represents an ego network for movies, categorized into two genres.
- **ENZYMES**: A biochemical dataset where each graph represents a protein, with nodes as amino acids and edges as spatial connections. The classification task categorizes enzymes into six classes.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/spiegel21/graph_nn.git
   cd graph_nn

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train and evaluate the models, use the following commands:

1. **Train a model**:
    ```bash
    python main.py
    ```

## Project Structure

- `data/`: Directory containing the datasets used for training and evaluation.
- `src/`: Source code for the project, including model definitions and utilities.
- `outputs/`: Directory where experiment results and logs are stored.
- `requirements.txt`: File listing the dependencies required for the project.
- `README.md`: This file, providing an overview and instructions for the project.

## Results

The results of the experiments, including performance metrics and visualizations, are documented in the `results/` directory. Detailed analysis and discussion of the findings can be found in the project report.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
