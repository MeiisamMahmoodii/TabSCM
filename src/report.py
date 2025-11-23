import os
import json
from src.visualize import plot_loss_curve, plot_gradient_norms

def generate_report(config):
    output_dir = config.get("output_dir", "causal_pfn_data")
    report_path = os.path.join(output_dir, "report.md")
    
    # Load Metrics
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    # Load Loss History
    loss_path = os.path.join(output_dir, "loss_history.json")
    loss_history = []
    if os.path.exists(loss_path):
        with open(loss_path, "r") as f:
            loss_history = json.load(f)
            
    # Generate Loss Plot
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    if loss_history:
        plot_loss_curve(loss_history, os.path.join(plots_dir, "loss_curve.png"))

    # Load and Plot Gradients
    grad_path = os.path.join(output_dir, "grad_history.json")
    grad_history = []
    if os.path.exists(grad_path):
        with open(grad_path, "r") as f:
            grad_history = json.load(f)
            
    if grad_history:
        plot_gradient_norms(grad_history, os.path.join(plots_dir, "gradient_flow.png"))
        
    # Create Markdown Content
    content = f"""# TabSCM Model Report

## 1. Model Architecture & Configuration
- **Max Columns**: {config.get('max_cols')}
- **Embedding Dimension**: {config.get('embed_dim')}
- **Training Steps**: {config.get('total_steps')}
- **Batch Size**: {config.get('batch_size')}
- **Learning Rate**: {config.get('lr')}

## 2. Training Performance
The model was trained for {config.get('total_steps')} steps.

### Loss Curve
![Loss Curve](plots/loss_curve.png)

### Gradient Flow
Analysis of gradient norms across model components.
![Gradient Flow](plots/gradient_flow.png)

## 3. Test Performance (Zero-Shot)
Evaluated on {config.get('n_test_samples')} unseen synthetic SCMs.

| Metric | Mean | Std Dev |
| :--- | :--- | :--- |
"""
    for k, v in metrics.items():
        if v['mean'] is not None:
            content += f"| **{k.upper()}** | {v['mean']:.4f} | {v['std']:.4f} |\n"
        else:
            content += f"| **{k.upper()}** | N/A | N/A |\n"
            
    content += """
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
"""

    with open(report_path, "w") as f:
        f.write(content)
        
    print(f"Report generated at {report_path}")
