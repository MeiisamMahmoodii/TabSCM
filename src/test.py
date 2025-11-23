import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.model import ZCIA_Transformer
from src.fast_generator import FastSCMGenerator
from src.visualize import plot_adjacency_comparison, plot_graph_structure, plot_metric_distributions
import os
import json

def compute_shd(target, pred):
    """Compute Structural Hamming Distance"""
    # pred is binary adjacency matrix
    diff = np.abs(target - pred)
    return np.sum(diff)

def test_model(config):
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")
    
    # Load Model
    output_dir = config.get("output_dir", "causal_pfn_data")
    model_path = os.path.join(output_dir, "model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return

    model = ZCIA_Transformer(
        max_cols=config["max_cols"],
        embed_dim=config["embed_dim"],
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 4)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    generator = FastSCMGenerator(
        min_nodes=config["min_nodes"],
        max_cols=config["max_cols"]
    )
    
    n_test_samples = config.get("n_test_samples", 100)
    metrics = {
        "shd": [],
        "auroc": [],
        "f1": [],
        "precision": [],
        "recall": []
    }
    
    print(f"Evaluating on {n_test_samples} samples...")
    
    with torch.no_grad():
        for idx in range(n_test_samples):
            data = generator.sample_scm()
            
            # Prepare input
            x = data['x'].unsqueeze(0).to(device) # Batch size 1
            m = data['m'].unsqueeze(0).to(device)
            y = data['y'].numpy() # Ground truth adjacency
            
            # Create pad mask (all false since we are not padding in this simple loop)
            # Note: If generator returns variable sizes, we might need to pad if batching.
            # Here we process one by one, so no padding needed for the model logic if it handles variable size correctly
            # BUT the model expects fixed size input if trained with padding.
            # The generator returns fixed size tensors (padded to max_cols)
            
            pad_mask = torch.zeros(1, config["max_cols"], dtype=torch.bool).to(device)
            # Adjust pad_mask based on actual nodes if generator didn't pad?
            # FastSCMGenerator returns padded tensors to max_cols?
            # Let's check FastSCMGenerator... it returns padded X if < max_cols.
            # But wait, FastSCMGenerator in src/fast_generator.py DOES NOT seem to pad in the return?
            # Actually, let's re-read src/fast_generator.py.
            # It stacks data_list. If X.shape[1] < max_cols... wait, the code I wrote for FastSCMGenerator
            # in the previous turn DOES NOT have the padding logic that was in CausalDataset.
            # It returns X_final directly.
            # However, ZCIA_Transformer expects inputs of shape (Batch, Rows, Cols).
            # If we run one by one, Cols varies.
            # The Transformer handles variable sequence length if we don't use pad_mask for attention?
            # Wait, the model uses `src_key_padding_mask=pad_mask`.
            # If we pass variable size to the model, it might fail if the model expects fixed embedding size?
            # No, the model projects (x, m) -> embed_dim.
            # Then transformer takes (Batch, Cols, Embed).
            # So variable Cols is fine for TransformerEncoder.
            # BUT the head outputs (Batch, Cols, Cols).
            # So we need to match y.
            
            # Let's check if we need to pad manually here or if the model handles it.
            # The model takes x: (B, R, C).
            # If C varies, it's fine.
            
            # However, we need to ensure y matches the output logits.
            # y from generator is (C, C).
            
            logits, _ = model(x, m, pad_mask[:, :x.shape[2]]) # Ignore intervention predictions
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)
            
            # Flatten for metrics
            y_flat = y.flatten()
            probs_flat = probs.flatten()
            pred_flat = (probs_flat > 0.3).astype(int)  # Lower threshold to 0.3
            
            # Compute Metrics
            metrics["shd"].append(compute_shd(y_flat, pred_flat))
            
            try:
                metrics["auroc"].append(roc_auc_score(y_flat, probs_flat))
            except ValueError:
                pass # Handle cases with only one class
                
            metrics["f1"].append(f1_score(y_flat, pred_flat))
            metrics["precision"].append(precision_score(y_flat, pred_flat, zero_division=0))
            metrics["recall"].append(recall_score(y_flat, pred_flat, zero_division=0))
            
            # Visualize first few samples
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            if idx < 3: # Plot first 3 samples
                plot_adjacency_comparison(y, probs, os.path.join(plots_dir, f"adj_comparison_{idx}.png"))
                plot_graph_structure(y, (probs > 0.5).astype(int), os.path.join(plots_dir, f"graph_structure_{idx}.png"))
            
    # Save metrics
    metrics_summary = {}
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        if v:
            mean_val = np.mean(v)
            std_val = np.std(v)
            metrics_summary[k] = {"mean": mean_val, "std": std_val}
            print(f"{k.upper()}: {mean_val:.4f} +/- {std_val:.4f}")
        else:
            metrics_summary[k] = {"mean": None, "std": None}
            print(f"{k.upper()}: N/A")
            
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_summary, f)
        
    # Plot metric distributions
    plot_metric_distributions(metrics, os.path.join(plots_dir, "metric_distributions.png"))
