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

def compute_masked_loss(logits, target, pad_mask):
    valid_cols = ~pad_mask 
    valid_matrix = torch.einsum('bi,bj->bij', valid_cols.float(), valid_cols.float())
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss_matrix = criterion(logits, target)
    masked_loss = loss_matrix * valid_matrix
    return masked_loss.sum() / (valid_matrix.sum() + 1e-6)

def train_model_online(config):
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = ZCIA_Transformer(
        max_cols=config["max_cols"],
        embed_dim=config["embed_dim"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    dataset = InfiniteCausalStream(
        min_nodes=config["min_nodes"],
        max_nodes=config["max_cols"]
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
        
        optimizer.zero_grad()
        
        logits = model(x, m, pad_mask)
        loss = compute_masked_loss(logits, y, pad_mask)
        
        loss.backward()
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

    return model
