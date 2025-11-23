import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.model import ZCIA_Transformer
from src.dataset import InfiniteCausalStream, causal_collate_fn

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)  # Probability of correct class
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

def compute_masked_loss(logits, target, pad_mask, loss_fn):
    """
    Compute loss only on valid (non-padded) entries.
    
    Args:
        logits: (batch, cols, cols) - predicted edge probabilities (logits)
        target: (batch, cols, cols) - ground truth adjacency matrix
        pad_mask: (batch, cols) - True for padded columns
        loss_fn: Loss function (FocalLoss or weighted BCE)
    """
    valid_cols = ~pad_mask 
    valid_matrix = torch.einsum('bi,bj->bij', valid_cols.float(), valid_cols.float())
    
    loss_matrix = loss_fn(logits, target)
    if len(loss_matrix.shape) == 0:  # Focal loss returns scalar
        return loss_matrix
    
    masked_loss = loss_matrix * valid_matrix
    return masked_loss.sum() / (valid_matrix.sum() + 1e-6)

def train_model_online(config):
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Initialize Focal Loss (more aggressive)
    focal_loss = FocalLoss(alpha=0.25, gamma=3.0)
    print(f"Using Focal Loss (alpha=0.25, gamma=3.0)")
    
    model = ZCIA_Transformer(
        max_cols=config["max_cols"],
        embed_dim=config["embed_dim"],
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 4)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    dataset = InfiniteCausalStream(
        min_nodes=config["min_nodes"],
        max_nodes=config["max_cols"],
        p_linear=config.get("p_linear", 0.5), # Pass p_linear to generator
        max_rows=config.get("max_rows", 1000) # Cap rows for memory
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        collate_fn=lambda b: causal_collate_fn(b, target_cols=config["max_cols"]),
        num_workers=0, # Set to 0 to avoid multiprocessing issues in some envs, or keep 4 if safe
        worker_init_fn=worker_init_fn
    )
    
    iterator = iter(dataloader)
    pbar = tqdm(range(config["total_steps"]))
    
    model.train()
    loss_history = []
    grad_history = []
    
    # Initialize Scaler for AMP (updated API)
    use_amp = config.get("use_amp", False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Gradient Accumulation
    accumulation_steps = config.get("accumulation_steps", 1)
    
    optimizer.zero_grad()
    
    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
            
        x = batch['x'].to(device)
        m = batch['m'].to(device)
        y = batch['y'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        
        # Forward pass with AMP (updated API)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, intervention_logits = model(x, m, pad_mask)
            
            # Main task: edge prediction
            edge_loss = compute_masked_loss(logits, y, pad_mask, focal_loss)
            
            # Auxiliary task: intervention prediction
            # Target: 1 if column has any intervention, 0 otherwise
            intervention_target = (m.sum(dim=1) > 0).float()  # (batch, cols)
            intervention_loss = F.binary_cross_entropy_with_logits(
                intervention_logits, 
                intervention_target,
                reduction='none'
            )
            # Mask out padded columns
            intervention_loss = (intervention_loss * (~pad_mask).float()).sum() / ((~pad_mask).float().sum() + 1e-6)
            
            # Combined loss
            loss = edge_loss + 0.1 * intervention_loss
            loss = loss / accumulation_steps # Normalize loss
        
        # Backward pass with Scaler
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # --- Log Gradients ---
        if step % config["print_every"] == 0:
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm().item()
                    # Group by component
                    if "set_encoder" in name:
                        key = "set_encoder"
                    elif "transformer" in name:
                        key = "transformer"
                    elif "bilinear" in name:
                        key = "head"
                    else:
                        key = "other"
                    
                    grad_norms.setdefault(key, []).append(norm)
            
            # Average norm per component
            avg_grads = {k: sum(v)/len(v) for k, v in grad_norms.items()}
            avg_grads["step"] = step
            grad_history.append(avg_grads)
        # ---------------------

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % config["print_every"] == 0:
            loss_val = loss.item()
            loss_history.append({"step": step, "loss": loss_val})
            pbar.set_description(f"Loss: {loss_val:.4f}")
            
    # Save the model
    output_dir = config.get("output_dir", "causal_pfn_data")
    import os
    import json
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    print(f"Model saved to {os.path.join(output_dir, 'model.pth')}")

    # Save loss history
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)
    print(f"Loss history saved to {os.path.join(output_dir, 'loss_history.json')}")

    # Save gradient history
    with open(os.path.join(output_dir, "grad_history.json"), "w") as f:
        json.dump(grad_history, f)
    print(f"Gradient history saved to {os.path.join(output_dir, 'grad_history.json')}")

    return model
