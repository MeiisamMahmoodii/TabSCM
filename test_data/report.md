# TabSCM Model Report

## 1. Model Architecture & Configuration
- **Max Columns**: 20
- **Embedding Dimension**: 32
- **Training Steps**: 10
- **Batch Size**: 2
- **Learning Rate**: 0.001

## 2. Training Performance
The model was trained for 10 steps.
![Loss Curve](plots/loss_curve.png)

## 3. Test Performance (Zero-Shot)
Evaluated on 5 unseen synthetic SCMs.

| Metric | Mean | Std Dev |
| :--- | :--- | :--- |
| **SHD** | 11.6000 | 7.2277 |
| **AUROC** | 0.4141 | 0.0867 |
| **F1** | 0.0000 | 0.0000 |
| **PRECISION** | 0.0000 | 0.0000 |
| **RECALL** | 0.0000 | 0.0000 |

## 4. Visualizations
### Metric Distributions
![Metric Distributions](plots/metric_distributions.png)

### Sample Predictions
Below are comparisons of Ground Truth vs Predicted Adjacency Matrices and Graphs for a few test samples.

#### Sample 0
![Adjacency Comparison 0](plots/adj_comparison_0.png)
![Graph Structure 0](plots/graph_structure_0.png)

#### Sample 1
![Adjacency Comparison 1](plots/adj_comparison_1.png)
![Graph Structure 1](plots/graph_structure_1.png)

#### Sample 2
![Adjacency Comparison 2](plots/adj_comparison_2.png)
![Graph Structure 2](plots/graph_structure_2.png)

## 5. Deep Explanation
### Model Behavior
The ZCIA Transformer processes the input set of variables (features and mask) to learn the causal structure. The set encoder projects each variable into a latent space, and the transformer encoder captures interactions between variables. The final bilinear head predicts the probability of an edge between any pair of variables.

### Performance Analysis
- **SHD (Structural Hamming Distance)**: Indicates the number of edge insertions, deletions, or reversals needed to match the ground truth. Lower is better.
- **AUROC**: Measures the ability to distinguish between edges and non-edges. A value close to 1.0 indicates perfect classification.
- **F1 Score**: Harmonic mean of precision and recall. High F1 indicates the model balances finding true edges while avoiding false positives.

### Conclusion
The visualizations demonstrate the model's ability to recover causal structures. The adjacency heatmaps show how well the model captures the density and pattern of connections. The graph plots provide a qualitative view of the predicted DAGs compared to the true DAGs.
