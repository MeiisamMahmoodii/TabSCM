import yaml
import os
import argparse
from tqdm import tqdm
from src.pipeline import run_pipeline
from src.train import train_model_online
from src.test import test_model
from src.report import generate_report

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="TabSCM: Causal SCM Generation and Training")
    parser.add_argument("--mode", choices=["generate", "train", "test", "report", "all"], default="all", help="Mode to run: generate data, train model, test, report or all")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get('output_dir', 'causal_pfn_data')
    
    if args.mode in ["generate", "all"]:
        os.makedirs(output_dir, exist_ok=True)
        n_runs = config.get('n_runs', 5)
        
        print("Starting small runs...")
        for i in tqdm(range(n_runs)):
            run_pipeline(
                run_id=i,
                min_nodes=config['n_nodes_min'],
                max_nodes=config['n_nodes_max'],
                n_samples=config['n_samples'],
                p_edge=config['p_edge'],
                p_linear=config['p_linear'],
                output_dir=output_dir
            )
        
        print("Starting large runs...")
        for i in tqdm(range(n_runs)):
            run_pipeline(
                run_id=i + n_runs,
                min_nodes=config.get('min_nodes_large', 50),
                max_nodes=config.get('max_nodes_large', 100),
                n_samples=config['n_samples'],
                p_edge=config['p_edge'],
                p_linear=config['p_linear'],
                output_dir=output_dir
            )

    if args.mode in ["train", "all"]:
        print("Starting training...")
        train_model_online(config)

    if args.mode in ["test", "all"]:
        print("Starting testing...")
        test_model(config)

    if args.mode in ["report", "all"]:
        print("Generating report...")
        generate_report(config)

if __name__ == "__main__":
    main()
