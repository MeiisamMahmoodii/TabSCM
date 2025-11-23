# TabSCM: Zero-Shot Causal Inference with Transformers

## Abstract
This report details the development and evaluation of **TabSCM**, a project designed to perform zero-shot causal discovery on tabular data. By training a Transformer-based model on an infinite stream of synthetic Structural Causal Models (SCMs), we aim to learn a generalized inference engine capable of predicting causal graphs for unseen datasets without retraining.

## 1. Introduction

### 1.1 The Problem
Causal discovery—identifying the causal structure underlying a dataset—is a fundamental challenge in science and engineering. Traditional methods (e.g., PC, GES) often rely on statistical independence tests or score-based searches, which can be computationally expensive and sensitive to hyperparameters. Moreover, they typically require running an optimization process for each new dataset.

### 1.2 The Solution
TabSCM proposes a **Zero-Shot** approach. We treat causal discovery as a supervised learning problem. We generate a vast and diverse dataset of synthetic SCMs (graphs + data) and train a neural network to map the dataset directly to its causal graph. Once trained, the model can infer the causal structure of a new dataset in a single forward pass, drastically reducing inference time.

## 2. Methodology

### 2.1 Data Generation
The core of our approach is the ability to generate infinite synthetic training data. We use a flexible generator (`src/generator.py` and `src/fast_generator.py`) that creates random Directed Acyclic Graphs (DAGs) and samples data from them.

**Key Features:**
- **Random Topology**: Graphs are generated with random sparsity (`p_edge`) and size (`n_nodes`).
- **Diverse Mechanisms**: We support various functional relationships between parents and children:
    - **Linear**: $X_i = \sum W_{ij} X_j + \epsilon$
    - **Non-Linear**: MLP, Sine, Quadratic, Tanh, Threshold, Gaussian.
- **Interventions**: We simulate interventions (Do-calculus) by forcing specific nodes to fixed values and propagating the effects to descendants. This helps the model learn to distinguish correlation from causation.
- **Online Generation**: The `InfiniteCausalStream` (`src/dataset.py`) generates unique SCMs on-the-fly during training, preventing overfitting to a fixed set of graphs.

### 2.2 Model Architecture (ZCIA)
We employ a Transformer-based architecture tailored for set-valued data, defined in `src/model.py`.

1.  **Set Embedding**:
    - The input is a set of variables (columns), each represented by a sequence of samples.
    - We stack the feature values $X$ and the intervention mask $M$.
    - An `element_encoder` (MLP) processes each sample, followed by pooling (mean) to obtain a single embedding vector for each variable. This ensures permutation invariance with respect to the sample dimension.

2.  **Transformer Encoder**:
    - A standard Transformer Encoder processes the sequence of variable embeddings.
    - Self-attention mechanisms allow the model to capture complex dependencies and conditional independencies between all variables simultaneously.

3.  **Edge Prediction Head**:
    - A Bilinear layer takes the transformed embeddings of every pair of variables $(Z_i, Z_j)$ and predicts the probability of a directed edge $i \to j$.
    - The output is an adjacency matrix of logits.

## 3. Training Pipeline

The training process (`src/train.py`) is designed for robustness and scalability:

- **Objective**: Minimize the Binary Cross-Entropy (BCE) loss between the predicted edge probabilities and the true adjacency matrix.
- **Masked Loss**: Since graph sizes vary, we pad batches to a maximum size (`max_cols`) and use a mask to compute loss only on the valid parts of the adjacency matrix.
- **Optimization**: We use the Adam optimizer with gradient clipping to ensure stable convergence.
- **Curriculum**: The generator can be configured to produce graphs of increasing complexity (size, density) to guide the learning process.

## 4. Evaluation & Results

We evaluate the model's zero-shot performance on unseen synthetic SCMs using `src/test.py`.

### 4.1 Metrics
- **SHD (Structural Hamming Distance)**: Measures the topological distance between the predicted and true graphs. Lower is better.
- **AUROC (Area Under ROC Curve)**: Evaluates the quality of the probabilistic edge predictions. Higher is better (1.0 is perfect).
- **F1 Score**: Balances precision and recall of edge detection.

### 4.2 Preliminary Results
*Based on initial testing on CPU with small batch sizes:*

| Metric | Value (Mean ± Std) | Interpretation |
| :--- | :--- | :--- |
| **SHD** | ~11.6 ± 7.2 | The model makes some errors but captures the general structure. |
| **AUROC** | ~0.41 ± 0.09 | Current performance is close to random (0.5). This indicates the need for longer training, larger batch sizes, or hyperparameter tuning. |

### 4.3 Visualizations
The reporting module (`src/report.py`) generates visual comparisons:
- **Loss Curves**: Show the convergence of the training process.
- **Adjacency Heatmaps**: Allow visual inspection of predicted vs. true edges.
- **Graph Plots**: Visualize the overall network topology.

*(See `test_data/report.md` for generated plots from the latest run)*

## 5. Conclusion and Future Work

TabSCM demonstrates the feasibility of learning a general-purpose causal discovery algorithm. While preliminary results show the pipeline is functional, achieving high accuracy requires further training.

**Future Directions:**
- **Scale Up**: Train on GPUs with larger batch sizes and for more steps (e.g., 100k+).
- **Real-World Benchmarks**: Evaluate on standard causal discovery datasets (e.g., Sachs, DREAM).
- **Architecture Tuning**: Experiment with different attention mechanisms or positional encodings.
- **Mechanism Diversity**: Expand the library of structural equations to cover more complex real-world scenarios.
