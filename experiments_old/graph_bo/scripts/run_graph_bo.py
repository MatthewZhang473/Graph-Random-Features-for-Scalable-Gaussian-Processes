"""
Script to run Bayesian Optimization experiments on social network datasets
using various algorithms including Random Search, BFS, DFS, GreedySearch and Sparse GRF.

Example usage:
    python run_graph_bo.py --config ../configs/default_config.yaml
    python run_graph_bo.py  # Uses default config
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'graph_bo'))
from graph_bo.data.database import GraphDataLoader
graph_data_loader = GraphDataLoader(data_root="graph_bo/data/raw_data", cache_dir="graph_bo/data/processed_data")
from graph_bo.utils import (
    RandomSearch, SparseGRF, BFS, DFS, GreedySearch, BayesianOptimizer,
    setup_gpytorch_settings, 
    save_results, print_summary, print_dataset_info, print_config, 
    convert_step_matrices_to_device, get_device, 
    get_step_matrices_scipy, load_config_from_yaml, get_default_config_path
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Graph Bayesian Optimization experiments')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file. Uses default config if not provided.')
    return parser.parse_args()

def run_experiment(dataset_name, algorithms, config):
    """Run BO experiment on a single dataset"""
    print(f"\n{'='*60}")
    print(f"🎯 Running BO experiments on {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    # Load dataset
    A, X, y = graph_data_loader(dataset_name)
    n_nodes = len(X)
    
    print_dataset_info(dataset_name, A, X, y)
    
    # Normalize targets, first take log, then gaussianize
    
    # ### Social Networks: Log + Gaussianize
    # y_logged = np.log(y+1e-6)
    # y_normalized = (y_logged - y_logged.mean()) / y_logged.std()
    # ### Wind Magnitude / Synthetic data: Just Gaussianize
    y_normalized = (y - y.mean()) / y.std()
    gt_best_value = float(y_normalized.max())

    device = get_device()
    print(f"  Device: {device}")
    
    # Get step matrices for GRF if needed
    step_matrices_scipy = get_step_matrices_scipy(
        dataset_name, A, config.step_matrices_dir,
        walks_per_node=config.walks_per_node,
        p_halt=config.p_halt,
        max_walk_length=config.max_walk_length,
        random_walk_seed=config.random_walk_seed
    )
    step_matrices_torch = convert_step_matrices_to_device(step_matrices_scipy, device)
    print(f"✅ Step matrices ready")
    
    all_results = []
    
    for algo_name in algorithms:
        print(f"\n🔬 Running {algo_name} with {len(config.seeds)} seeds...")
        
        for bo_seed_idx, bo_seed in enumerate(config.seeds):
            print(f"   Seed {bo_seed_idx + 1}/{len(config.seeds)} (seed={bo_seed})")
            
            # Create algorithm
            if algo_name == 'random_search':
                algorithm = RandomSearch(n_nodes, device)
            elif algo_name == 'greedy_search':
                algorithm = GreedySearch(A, n_nodes, device)
            elif algo_name == 'bfs':
                algorithm = BFS(A, n_nodes, device)
            elif algo_name == 'dfs':
                algorithm = DFS(A, n_nodes, device)
            elif algo_name == 'sparse_grf':
                algorithm = SparseGRF(
                    n_nodes, device, step_matrices_torch,
                    config.max_walk_length, config.learning_rate,
                    config.train_epochs, config.gp_retrain_interval
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo_name}")
            
            # Reset algorithm state
            if hasattr(algorithm, 'reset_cache'):
                algorithm.reset_cache()
            
            # Run BO
            optimizer = BayesianOptimizer(
                algorithm, y_normalized, 
                config.initial_points, config.batch_size
            )
            
            results = optimizer.run_optimization(
                config.iterations, 
                seed=bo_seed, 
                algorithm_name=algo_name.replace('_', ' ').title()
            )
            
            # Add metadata to results
            for result in results:
                result.update({
                    'algorithm': algo_name,
                    'dataset': dataset_name,
                    'bo_seed': bo_seed,
                    'bo_run': bo_seed_idx + 1,
                    'ground_truth_best': gt_best_value,
                    'n_nodes': n_nodes,
                    'n_edges': A.nnz // 2,
                    'density': A.nnz / (A.shape[0] * A.shape[1])
                })
            
            all_results.extend(results)
            
            # Cleanup
            del algorithm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return pd.DataFrame(all_results)

def main():
    args = parse_args()
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = get_default_config_path()
        print(f"📋 Using default config: {config_path}")
    
    config = load_config_from_yaml(config_path)
    print(f"📋 Loaded configuration from: {config_path}")
    
    # Setup random seeds
    setup_gpytorch_settings()
    np.random.seed(config.numpy_seed)
    torch.manual_seed(config.torch_seed)
    
    # Print configuration summary
    config_dict = {
        'datasets': config.datasets,
        'algorithms': config.algorithms,
        'iterations': config.iterations,
        'runs': config.runs,
        'initial_points': config.initial_points,
        'batch_size': config.batch_size,
        'seeds': config.seeds
    }
    print_config(config_dict)
    
    # Run experiments
    all_results = []
    for dataset in config.datasets:
        try:
            results_df = run_experiment(dataset, config.algorithms, config)
            all_results.append(results_df)
        except Exception as e:
            print(f"❌ Error with dataset {dataset}: {e}")
    
    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save and summarize
        dataset_suffix = "_".join(config.datasets) if len(config.datasets) > 1 else config.datasets[0]
        save_results(combined_results, config.results_dir, suffix=f"_{dataset_suffix}")
        print_summary(combined_results)
        
        print(f"\n✅ All experiments completed!")
    else:
        print("❌ No experiments completed successfully")

if __name__ == "__main__":
    main()