import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import os

def plot_loss_curve(loss_history, save_path):
    steps = [entry['step'] for entry in loss_history]
    losses = [entry['loss'] for entry in loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_adjacency_comparison(target, pred, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(target, ax=axes[0], cmap="Greys", cbar=False)
    axes[0].set_title("Ground Truth")
    
    sns.heatmap(pred, ax=axes[1], cmap="Greys", cbar=False)
    axes[1].set_title("Predicted")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_graph_structure(target, pred, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    G_target = nx.from_numpy_array(target, create_using=nx.DiGraph)
    G_pred = nx.from_numpy_array(pred, create_using=nx.DiGraph)
    
    pos = nx.spring_layout(G_target, seed=42) # Use same layout if possible, or recompute
    
    nx.draw(G_target, pos, ax=axes[0], with_labels=True, node_color='lightblue', edge_color='black', arrowsize=15)
    axes[0].set_title("Ground Truth Graph")
    
    # For prediction, we might want to use the same pos if nodes match, but let's recompute for now or use same
    # If nodes are same indices, use same pos
    nx.draw(G_pred, pos, ax=axes[1], with_labels=True, node_color='lightgreen', edge_color='black', arrowsize=15)
    axes[1].set_title("Predicted Graph")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metric_distributions(metrics, save_path):
    # metrics is a dict of lists: {'shd': [...], 'auroc': [...], ...}
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
        
    for i, (name, values) in enumerate(metrics.items()):
        if not values:
            continue
        sns.histplot(values, ax=axes[i], kde=True)
        axes[i].set_title(f"{name.upper()} Distribution")
        axes[i].set_xlabel(name.upper())
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_gradient_norms(grad_history, save_path):
    steps = [entry['step'] for entry in grad_history]
    
    plt.figure(figsize=(10, 6))
    
    # Collect keys (components)
    keys = set()
    for entry in grad_history:
        keys.update(entry.keys())
    keys.discard('step')
    
    for key in keys:
        values = [entry.get(key, 0) for entry in grad_history]
        plt.plot(steps, values, label=key)
        
    plt.xlabel('Step')
    plt.ylabel('Average Gradient Norm')
    plt.title('Gradient Flow Analysis')
    plt.legend()
    plt.yscale('log') # Log scale is often better for gradients
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(save_path)
    plt.close()
