import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import numpy as np
import os
import glob
from src.fast_generator import FastSCMGenerator

class CausalDataset(Dataset):
    def __init__(self, root_dir, max_rows=1000, max_cols=128, clip_mag=1e6):
        """
        root_dir: The folder where you saved 'run_0', 'run_1', etc.
        max_rows: cap number of rows sampled from the stacked dataset
        max_cols: cap number of columns (variables) per dataset
        clip_mag: maximum absolute value to clip numeric columns to (prevents overflow)
        """
        self.run_dirs = sorted(glob.glob(os.path.join(root_dir, "run_*")))
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.clip_mag = clip_mag

    def __len__(self):
        return len(self.run_dirs)

    def __getitem__(self, idx):
        run_path = self.run_dirs[idx]

        # --- STEP 1: LOADING ---
        obs_df = pd.read_csv(os.path.join(run_path, "observational_data.csv"))
        obs_df = obs_df.apply(pd.to_numeric, errors='coerce')

        obs_col_means = obs_df.mean(axis=0)
        obs_col_means = obs_col_means.fillna(0.0)

        obs_filled = obs_df.fillna(obs_col_means)
        obs_vals = obs_filled.values.astype(np.float64)
        obs_vals = np.clip(obs_vals, -self.clip_mag, self.clip_mag)
        obs_vals = np.nan_to_num(obs_vals, nan=0.0, posinf=self.clip_mag, neginf=-self.clip_mag).astype(np.float32)

        int_dir = os.path.join(run_path, "interventional_data")
        int_files = []
        if os.path.exists(int_dir):
            int_files = glob.glob(os.path.join(int_dir, "do_X*.csv"))

        data_list = [obs_vals]
        mask_list = [np.zeros_like(obs_vals, dtype=float)]

        # --- STEP 2: MASKING ---
        for f in int_files:
            int_df = pd.read_csv(f)
            filename = os.path.basename(f)
            try:
                col_name = filename.split("_")[1].split(".")[0]
            except Exception:
                col_name = None

            reindexed = int_df.reindex(columns=obs_df.columns)
            reindexed = reindexed.apply(pd.to_numeric, errors='coerce')
            reindexed = reindexed.replace([np.inf, -np.inf], np.nan)
            reindexed = reindexed.fillna(obs_col_means)

            data_vals = reindexed.values.astype(np.float64)
            col_means_arr = obs_col_means.values.astype(np.float64)
            mask_finite = np.isfinite(data_vals)
            if not np.all(mask_finite):
                rows, cols = data_vals.shape
                for c in range(cols):
                    col_finite = mask_finite[:, c]
                    if not np.all(col_finite):
                        data_vals[~col_finite, c] = col_means_arr[c]

            data_vals = np.clip(data_vals, -self.clip_mag, self.clip_mag)
            data_vals = data_vals.astype(np.float32)

            m = np.zeros_like(data_vals, dtype=float)

            if col_name and col_name in obs_df.columns:
                col_idx = obs_df.columns.get_loc(col_name)
                m[:, col_idx] = 1.0

            data_list.append(data_vals)
            mask_list.append(m)

        X = np.vstack(data_list).astype(np.float32)
        M = np.vstack(mask_list).astype(np.float32)

        if X.shape[1] > self.max_cols:
            X = X[:, :self.max_cols]
            M = M[:, :self.max_cols]

        if X.shape[1] < self.max_cols:
            n_missing = self.max_cols - X.shape[1]
            X = np.pad(X, ((0, 0), (0, n_missing)), mode='constant', constant_values=0.0)
            M = np.pad(M, ((0, 0), (0, n_missing)), mode='constant', constant_values=0.0)

        if X.shape[0] > self.max_rows:
            indices = np.random.choice(X.shape[0], self.max_rows, replace=False)
            X = X[indices]
            M = M[indices]

        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-5
        X = (X - mean) / std

        adj = np.load(os.path.join(run_path, "adjacency_matrix.npy"))
        if adj.shape[0] > self.max_cols or adj.shape[1] > self.max_cols:
            adj = adj[:self.max_cols, :self.max_cols]
        if adj.shape[0] < self.max_cols or adj.shape[1] < self.max_cols:
            a = np.zeros((self.max_cols, self.max_cols), dtype=adj.dtype)
            r = min(adj.shape[0], self.max_cols)
            c = min(adj.shape[1], self.max_cols)
            a[:r, :c] = adj[:r, :c]
            adj = a

        return {
            "x": torch.tensor(X), 
            "m": torch.tensor(M), 
            "y": torch.tensor(adj).float()
        }

class InfiniteCausalStream(IterableDataset):
    def __init__(self, min_nodes=8, max_nodes=128, p_linear=0.5, max_rows=1000):
        self.generator = FastSCMGenerator(min_nodes=min_nodes, max_cols=max_nodes, p_linear=p_linear, max_rows=max_rows)

    def __iter__(self):
        while True:
            try:
                data = self.generator.sample_scm()
                yield data
            except Exception as e:
                continue

def causal_collate_fn(batch, target_cols=128):
    observed_max_cols = max([item['x'].shape[1] for item in batch])
    max_cols = min(target_cols, observed_max_cols)

    batch_size = len(batch)
    max_rows = max([item['x'].shape[0] for item in batch])

    x_padded = torch.zeros(batch_size, max_rows, target_cols)
    m_padded = torch.zeros(batch_size, max_rows, target_cols)
    y_padded = torch.zeros(batch_size, target_cols, target_cols)

    pad_mask = torch.ones(batch_size, target_cols, dtype=torch.bool)

    for i, item in enumerate(batch):
        rows = item['x'].shape[0]
        d = item['x'].shape[1]
        d_trunc = min(d, target_cols)

        x_padded[i, :rows, :d_trunc] = item['x'][:, :d_trunc]
        m_padded[i, :rows, :d_trunc] = item['m'][:, :d_trunc]
        y_padded[i, :d_trunc, :d_trunc] = item['y'][:d_trunc, :d_trunc]

        pad_mask[i, :d_trunc] = False 

    return {
        "x": x_padded, 
        "m": m_padded, 
        "y": y_padded, 
        "pad_mask": pad_mask
    }
