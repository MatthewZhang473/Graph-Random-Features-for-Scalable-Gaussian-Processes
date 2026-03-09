"""
Bayesian Optimization utilities for Graph GP experiments
"""
import os, json
from datetime import datetime
from .config import setup_gpytorch_settings, create_directories
from .data_utils import generate_grid_data, get_cached_data, get_step_matrices, convert_to_device
from .gp_models import SparseGraphGP
from .bo_algorithms import Algorithm, RandomSearch, SparseGRF, BFS, DFS, BayesianOptimizer
from .io_utils import save_results, print_summary

__all__ = [
    'setup_gpytorch_settings', 'create_directories',
    'generate_grid_data', 'get_cached_data', 'get_step_matrices', 'convert_to_device',
    'SparseGraphGP',
    'Algorithm', 'RandomSearch', 'SparseGRF', 'BFS', 'DFS', 'BayesianOptimizer',
    'save_results', 'print_summary'
]

def save_results(results_df, config, suffix=""):
    """Save BO results with timestamped files and configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Include suffix (e.g., graph type) in filename
    base_name = f'bo_experiment_results{suffix}'
    main_file = os.path.join(config.RESULTS_DIR, f'{base_name}.csv')
    timestamped_file = os.path.join(config.RESULTS_DIR, f'{base_name}_{timestamp}.csv')
    
    results_df.to_csv(main_file, index=False)
    results_df.to_csv(timestamped_file, index=False)
    
    # Save configuration with graph type info
    config_summary = {
        'timestamp': timestamp,
        'graph_type': getattr(config, 'GRAPH_TYPE', 'unknown'),
        'total_experiments': len(results_df),
        'n_nodes': config.N_NODES,
        'n_edges': getattr(config, 'N_EDGES', None),
        'avg_degree': getattr(config, 'AVG_DEGREE', None),
        'n_peaks': getattr(config, 'N_PEAKS', None),
    }
    
    config_file = os.path.join(config.RESULTS_DIR, f'bo_experiment_config{suffix}_{timestamp}.json')
    with open(config_file, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"📁 Results saved with graph type {getattr(config, 'GRAPH_TYPE', 'unknown')}:")
    print(f"   Main: {main_file}")
    print(f"   Config: {config_file}")
    
    return {'main_file': main_file, 'config_file': config_file}
