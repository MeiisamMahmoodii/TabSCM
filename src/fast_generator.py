import numpy as np
import networkx as nx
import torch

class FastSCMGenerator:
    def __init__(self, min_nodes=8, max_cols=128, max_rows=2000):
        self.min_nodes = min_nodes
        self.max_cols = max_cols 
        self.max_rows = max_rows # New parameter to cap memory usage
        
    def sample_scm(self):
        """Creates one random SCM and returns formatted tensors directly."""
        # 1. Randomly determine graph size 
        n_nodes = np.random.randint(self.min_nodes, self.max_cols + 1)
        
        # 2. Build Random DAG
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n_nodes))
        p_edge = np.random.uniform(0.1, 0.3)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < p_edge:
                    graph.add_edge(i, j)
                    
        # 3. Generate Observational Data
        n_samples = 1000 # Base pool size
        X = np.zeros((n_samples, n_nodes), dtype=np.float32)
        node_types = np.random.randint(0, 2, size=n_nodes) 
        
        topo_order = list(nx.topological_sort(graph))
        
        for node in topo_order:
            parents = list(graph.predecessors(node))
            noise = np.random.normal(0, 0.1, size=n_samples)
            
            if len(parents) == 0:
                X[:, node] = np.random.normal(0, 1, size=n_samples)
            else:
                parent_data = X[:, parents]
                if node_types[node] == 0: 
                    weights = np.random.uniform(-2, 2, size=len(parents))
                    X[:, node] = np.dot(parent_data, weights) + noise
                else: 
                    weights = np.random.uniform(-2, 2, size=len(parents))
                    X[:, node] = np.tanh(np.dot(parent_data, weights)) + noise

        # 4. Generate Interventions
        n_interventions = max(1, int(n_nodes * 0.2))
        nodes_to_intervene = np.random.choice(range(n_nodes), n_interventions, replace=False)
        
        data_list = [X]
        mask_list = [np.zeros_like(X)]
        
        for target_node in nodes_to_intervene:
            X_int = X.copy()
            intervene_val = np.random.uniform(3, 5)
            X_int[:, target_node] = intervene_val
            
            descendants = nx.descendants(graph, target_node)
            for node in topo_order:
                if node in descendants:
                    parents = list(graph.predecessors(node))
                    parent_data = X_int[:, parents]
                    noise = np.random.normal(0, 0.1, size=n_samples)
                    if node_types[node] == 0:
                        weights = np.random.uniform(-2, 2, size=len(parents))
                        X_int[:, node] = np.dot(parent_data, weights) + noise
                    else:
                        weights = np.random.uniform(-2, 2, size=len(parents))
                        X_int[:, node] = np.tanh(np.dot(parent_data, weights)) + noise
            
            M_int = np.zeros_like(X_int)
            M_int[:, target_node] = 1.0
            
            data_list.append(X_int)
            mask_list.append(M_int)
            
        # 5. Stack 
        X_final = np.vstack(data_list)
        M_final = np.vstack(mask_list)
        
        # --- CRITICAL FIX: Subsample Rows to prevent OOM ---
        if X_final.shape[0] > self.max_rows:
            indices = np.random.choice(X_final.shape[0], self.max_rows, replace=False)
            X_final = X_final[indices]
            M_final = M_final[indices]
        # ---------------------------------------------------

        # Normalize
        mean = X_final.mean(axis=0)
        std = X_final.std(axis=0) + 1e-5
        X_final = (X_final - mean) / std
        
        # 6. Get Adjacency Matrix
        adj = nx.to_numpy_array(graph)
        
        return {
            "x": torch.tensor(X_final, dtype=torch.float32),
            "m": torch.tensor(M_final, dtype=torch.float32),
            "y": torch.tensor(adj, dtype=torch.float32)
        }
