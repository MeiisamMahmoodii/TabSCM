import numpy as np
from src.generator import CausalSCMGenarator

def run_pipeline(run_id, min_nodes, max_nodes, n_samples, p_edge, p_linear, output_dir):
    n_nodes = np.random.randint(min_nodes, max_nodes)
    scm = CausalSCMGenarator(n_nodes=n_nodes, p_edge=p_edge, p_linear=p_linear)
    scm._build_structure()
    scm._assign_mechanisms()
    df_obs = scm.generate_data(n_samples=n_samples)
    number_of_interventions = max(1, n_nodes // 5)
    nodes_to_intervene = np.random.choice(range(n_nodes), size=number_of_interventions, replace=False)
    intervention_datasets = {}

    for node in nodes_to_intervene:
        intervene_val = 5.0
        df_int = scm.generate_data(n_samples=n_samples//5, interventions={node: intervene_val})
        intervention_datasets[node] = df_int

    scm.save_artifacts(run_id, df_obs, intervention_datasets, output_dir)
