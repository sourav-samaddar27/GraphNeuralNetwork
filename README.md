Predictive Drug Discovery using Graph Neural Networks
This project implements a Graph Attention Network (GAT) to predict the bioactivity of small molecules, a critical task in early-stage drug discovery. The model is trained on the BACE (β-Secretase 1) dataset to classify compounds based on their potential to inhibit the BACE-1 enzyme, a key target in Alzheimer's disease research.

📜 Project Overview
The goal of this project is to demonstrate the power of Graph Neural Networks (GNNs) in understanding and predicting molecular properties. Traditional machine learning models often rely on manually engineered features (molecular fingerprints), which can be limiting. GNNs, however, learn directly from the graph structure of a molecule (atoms as nodes, bonds as edges), allowing them to capture more nuanced and powerful representations.

This repository provides a complete end-to-end pipeline:

Data loading and preprocessing for molecular graphs.

Implementation of an enhanced GAT model with modern deep learning techniques.

A robust training and evaluation framework.

Handling of common real-world data challenges like class imbalance.

✨ Key Features
Graph Attention Network (GAT): Built a sophisticated GNN that uses attention mechanisms to weigh the importance of different atoms within a molecule.

Enhanced Architecture: Integrated Batch Normalization for stable training and a dual-pooling mechanism (concatenating mean and add pooling) for richer graph-level embeddings.

Class Imbalance Handling: Implemented a weighted loss function (BCEWithLogitsLoss with pos_weight) to effectively train on the imbalanced BACE dataset.

Performance Optimization: Utilized an AdamW optimizer and a ReduceLROnPlateau learning rate scheduler to achieve robust convergence and prevent overfitting.

📊 Performance
The model was trained and evaluated on the BACE dataset, achieving strong predictive performance on the unseen test set.

Test Accuracy: [Your-Best-Accuracy]% (e.g., 84.1%)

Test ROC-AUC Score: [Your-Best-ROC-AUC] (e.g., 0.88)

🛠️ Technologies & Libraries
Python 3.x

PyTorch: The core deep learning framework.

PyTorch Geometric (PyG): The essential library for building GNNs in PyTorch.

RDKit: Used by PyG for cheminformatics and converting molecules to graphs.

Scikit-learn: For performance metrics (ROC-AUC score).

Matplotlib: For plotting training results.

🚀 Setup and Installation
To run this project, you need to have Python and PyTorch installed. Then, you can install the required dependencies.

1. Clone the repository:

git clone [your-github-repository-link]
cd [repository-folder-name]

2. Install PyTorch:
Follow the instructions on the official PyTorch website for your specific system (CPU/GPU). An example for a CPU-only version is:

pip install torch

3. Install PyTorch Geometric and other dependencies:
PyG requires specific packages that are compatible with your PyTorch version. The following command is recommended for a robust installation.

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.1.0+cpu.html](https://data.pyg.org/whl/torch-2.1.0+cpu.html)
pip install torch_geometric
pip install scikit-learn matplotlib

(Note: The PyG installation command might change. Always check the official documentation for the latest instructions.)

⚙️ How to Run
The entire pipeline is contained within a single Python script. Simply run it from your terminal:

python your_script_name.py

The script will automatically download the BACE dataset, preprocess it, build the model, and start the training and evaluation process. The final results and plots will be displayed upon completion.

🔮 Future Improvements
Hyperparameter Tuning: Use a systematic approach like Optuna or Ray Tune to find the optimal set of hyperparameters (learning rate, hidden dimensions, number of heads, etc.).

Try Different GNN Architectures: Implement and compare the performance of other GNN layers like GINConv or GCNConv.

Explore Other Datasets: Adapt the pipeline to other molecular prediction tasks from the MoleculeNet suite, such as ClinTox (toxicity) or HIV (antiviral activity).
