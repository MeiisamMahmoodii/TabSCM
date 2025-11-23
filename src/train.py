import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.model import ZCIA_Transformer
from src.dataset import InfiniteCausalStream, causal_collate_fn

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def compute_masked_loss(logits, target, pad_mask, pos_weight=None):
    valid_cols = ~pad_mask 
    valid_matrix = torch.einsum('bi,bj->bij', valid_cols.float(), valid_cols.float())
    
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
    loss_matrix = criterion(logits, target)
    masked_loss = loss_matrix * valid_matrix
    return masked_loss.sum() / (valid_matrix.sum() + 1e-6)

def train_model_online(config):
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Calculate pos_weight for weighted loss
    p_edge = config.get("p_edge", 0.3)
    # Weight = (Negative Samples) / (Positive Samples) approx (1 - p) / p
    # We can boost it slightly to prioritize recall
    weight_val = (1.0 - p_edge) / p_edge
    pos_weight = torch.tensor([weight_val]).to(device)
    print(f"Using weighted loss with pos_weight: {weight_val:.2f}")
    
    model = ZCIA_Transformer(
        max_cols=config["max_cols"],
        embed_dim=config["embed_dim"]
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
    
    # Initialize Scaler for AMP
    use_amp = config.get("use_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Gradient Accumulation
    accumulation_steps = config.get("accumulation_steps", 1)
    
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
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x, m, pad_mask)
            loss = compute_masked_loss(logits, y, pad_mask, pos_weight=pos_weight)
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
