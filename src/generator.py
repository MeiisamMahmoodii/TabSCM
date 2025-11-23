import numpy as np
import pandas as pd
import networkx as nx
import os
import json
from src.mechanism import TabPFNMechanism

class CausalSCMGenarator:
    def __init__(self, n_nodes, p_edge, p_linear):
        self.n_nodes = n_nodes
        self.graph = nx.DiGraph()
        self.mechanisms = {}
        self.node_types = {}
        self.p_edge = p_edge
        self.p_linear = p_linear

    def _build_structure(self):
        nodes = range(self.n_nodes)
        self.graph.add_nodes_from(nodes)
        for i in nodes:
            for j in range(i+1, self.n_nodes):
                if np.random.rand() < self.p_edge:
                    self.graph.add_edge(j, i)

    def _assign_mechanisms(self):
        nonlinear_types = ['mlp', 'sin', 'quadratic', 'threshold', 'gaussian']
        for node in self.graph.nodes:
            parents = list(self.graph.predecessors(node))
            is_linear = np.random.rand() < self.p_linear

            if is_linear:
                self.node_types[node] = 'linear'
            else:
                mech_type = np.random.choice(nonlinear_types)
                self.node_types[node] = mech_type

            if len(parents) == 0:
                self.mechanisms[node] = None
            else:
                if self.node_types[node] == 'linear':
                    weights = np.random.uniform(-2, 2, size=len(parents))
                    self.mechanisms[node] = weights
                elif self.node_types[node] == 'mlp':
                    mlp = TabPFNMechanism(input_dim=len(parents))
                    self.mechanisms[node] = mlp
                elif self.node_types[node] == 'sin':
                    w = np.random.uniform(-2, 2, size=len(parents))
                    b = np.random.uniform(-1, 1)
                    self.mechanisms[node] = lambda X, w=w, b=b: np.sin(np.dot(X, w) + b)
                elif self.node_types[node] == 'quadratic':
                    a = np.random.uniform(-1, 1, size=len(parents))
                    c = np.random.uniform(-1, 1)
                    self.mechanisms[node] = lambda X, a=a, c=c: (np.dot(X, a) + c) ** 2
                elif self.node_types[node] == 'threshold':
                    w = np.random.uniform(-2, 2, size=len(parents))
                    th = np.random.uniform(-0.5, 0.5)
                    self.mechanisms[node] = lambda X, w=w, th=th: (np.dot(X, w) > th).astype(float)
                elif self.node_types[node] == 'gaussian':
                    w = np.random.uniform(-2, 2, size=len(parents))
                    b = np.random.uniform(-1, 1)
                    sigma = np.random.uniform(0.1, 1.0)
                    self.mechanisms[node] = lambda X, w=w, b=b, s=sigma: np.exp(-((np.dot(X, w) + b) ** 2) / (2 * s ** 2))
                else:
                    self.mechanisms[node] = TabPFNMechanism(input_dim=len(parents))

    def generate_data(self, n_samples, interventions=None):
        data = np.zeros((n_samples, self.n_nodes))
        topo_order = list(nx.topological_sort(self.graph))

        for node in topo_order:
            if interventions and node in interventions:
                data[:, node] = interventions[node]
                continue

            parents = list(self.graph.predecessors(node))
            epsilon = np.random.normal(0, 0.1, size=n_samples)

            if len(parents) == 0:
                data[:, node] = np.random.normal(0, 1, size=n_samples)
            else:
                parent_data = data[:, parents]
                if self.node_types[node] == 'linear':
                    weights = self.mechanisms[node]
                    effect = np.dot(parent_data, weights)
                else:
                    mech = self.mechanisms[node]
                    if isinstance(mech, TabPFNMechanism):
                         effect = mech(parent_data)
                    else:
                         effect = mech(parent_data)
                data[:, node] = effect + epsilon
        return pd.DataFrame(data, columns=[f'X{i}' for i in range(self.n_nodes)])

    def save_artifacts(self, run_id, obs_df, interventional_dfs, output_dir):
        run_dir = os.path.join(output_dir, f'run_{run_id}')
        os.makedirs(run_dir, exist_ok=True)

        obs_df.to_csv(os.path.join(run_dir, 'observational_data.csv'), index=False)

        int_dir = os.path.join(run_dir, 'interventional_data')
        os.makedirs(int_dir, exist_ok=True)
        for intervention, df in interventional_dfs.items():
            df.to_csv(os.path.join(int_dir, f'do_X{intervention}.csv'), index=False)

        adj_matrix = nx.to_numpy_array(self.graph)
        np.save(os.path.join(run_dir, 'adjacency_matrix.npy'), adj_matrix)

        with open(os.path.join(run_dir, 'mechanisms.json'), 'w') as f:
            nodes = [int(n) for n in sorted(self.graph.nodes)]
            parents_map = {str(int(n)): [int(p) for p in sorted(self.graph.predecessors(n))] for n in self.graph.nodes}
            edges = {}
            mechanisms_summary = {}
            for node in self.graph.nodes:
                mech = self.mechanisms.get(node)
                mech_type = self.node_types.get(node)
                mechanisms_summary[str(int(node))] = mech_type
                parents = list(self.graph.predecessors(node))
                if mech_type == 'linear' and mech is not None:
                    weights = np.array(mech).tolist() if hasattr(mech, 'tolist') or isinstance(mech, np.ndarray) else list(mech)
                    mechanisms_summary[str(int(node)) + '_weights'] = [float(w) for w in weights]
                    for idx, p in enumerate(parents):
                        edges[f'{int(p)}->{int(node)}'] = {'type': 'linear', 'weight': float(weights[idx])}
                elif mech_type == 'mlp':
                    for p in parents:
                        edges[f'{int(p)}->{int(node)}'] = {'type': 'mlp'}
                else:
                    for p in parents:
                        edges[f'{int(p)}->{int(node)}'] = {'type': mech_type}
            intervened_nodes = [int(k) for k in interventional_dfs.keys()]
            json.dump({
                'nodes': nodes,
                'parents': parents_map,
                'edges': edges,
                'mechanisms': mechanisms_summary,
                'intervened_nodes': intervened_nodes,
                'adjacency_matrix_shape': list(adj_matrix.shape)
            }, f, indent=4)

        print(f"Saved artifacts for run {run_id}:{self.n_nodes} nodes, {len(interventional_dfs)} interventions.")
