import os
import pandas as pd
from typing import Dict, Any
import pickle
import hashlib
import scipy.sparse as sp
from typing import List
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)
from efficient_graph_gp_sparse.preprocessor import GraphPreprocessor

# 1. Utility function for generating paths (for step matrices)
def get_step_matrices_file_name(base_dir: str,
                dataset: str,
                walks_per_node: int,
                p_halt: float,
                max_walk_length: int,
                random_walk_seed: int) -> str:
    """Create a unique path for storing step matrices."""
    
    dataset_dir = os.path.join(base_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    return os.path.join(
        dataset_dir,
        f"step_matrices_walks_{walks_per_node}_halt_{p_halt}_maxlen_{max_walk_length}_seed_{random_walk_seed}.pkl"
    )

# 2. Functions for loading and saving step matrices

def get_step_matrices_scipy(dataset_name: str,
                            adj_matrix: sp.csr_matrix,
                            base_dir: str,
                            walks_per_node: int=100,
                            p_halt: float=0.1,
                            max_walk_length: int=4,
                            random_walk_seed: int=42) -> List[sp.csr_matrix]:
    """
    Load step matrices from cache or compute them using GraphPreprocessor.
    
    Args:
        adj_matrix: Adjacency matrix
        walks_per_node: Number of walks per node
        p_halt: Halt probability
        max_walk_length: Maximum walk length
        cache_dir: Cache directory
        dataset_name: Dataset name for logging
        
    Returns:
        List of CSR sparse matrices (scipy) representing step matrices
    """

    # Create cache filename in the specified directory
    os.makedirs(base_dir, exist_ok=True)
    cache_filename = get_step_matrices_file_name(base_dir, dataset_name, walks_per_node, p_halt, max_walk_length, random_walk_seed)
    
    # Check if cached matrices exist
    load_from_disk = os.path.exists(cache_filename)
    
    if load_from_disk:
        print(f"Loading step matrices from cache: {os.path.basename(cache_filename)}")
    else:
        print(f"Computing step matrices for {dataset_name}...")
    
    # Create GraphPreprocessor instance
    preprocessor = GraphPreprocessor(
        adjacency_matrix=adj_matrix,
        walks_per_node=walks_per_node,
        p_halt=p_halt,
        max_walk_length=max_walk_length,
        random_walk_seed=random_walk_seed,
        load_from_disk=load_from_disk,
        use_tqdm=True,
        cache_filename=cache_filename,
        n_processes=None  # Use all available cores
    )
    
    if not load_from_disk:
        # Compute and save step matrices
        _ = preprocessor.preprocess_graph(save_to_disk=True)
        print(f"Cached step matrices to: {os.path.basename(cache_filename)}")

    # Return the scipy CSR matrices
    return preprocessor.step_matrices_scipy

# 3. Functions for analyzing and saving experiment results

def save_results(results_df: pd.DataFrame, results_dir: str, suffix: str = "") -> str:
    """
    Save experiment results to CSV files.
    
    Args:
        results_df: DataFrame containing experiment results
        results_dir: Directory to save results
        suffix: Optional suffix for filename
        
    Returns:
        Path to saved results file
    """
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bo_results_{timestamp}{suffix}.csv"
    filepath = os.path.join(results_dir, filename)
    
    # Save main results
    results_df.to_csv(filepath, index=False)
    print(f"💾 Results saved to: {filepath}")
    
    # Save summary statistics
    summary = compute_summary_stats(results_df)
    summary_file = filepath.replace('.csv', '_summary.csv')
    summary.to_csv(summary_file)
    print(f"📊 Summary saved to: {summary_file}")
    
    return filepath

def compute_summary_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for experiment results."""
    return results_df.groupby(['dataset', 'algorithm']).agg({
        'best_value': ['mean', 'std', 'max'],
        'regret': ['mean', 'std', 'min'],
        'iteration': 'count'
    }).round(4)

def print_summary(results_df: pd.DataFrame) -> None:
    """Print experiment summary to console."""
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    final_results = results_df[results_df['iteration'] == results_df['iteration'].max()]
    
    for dataset in results_df['dataset'].unique():
        print(f"\n{dataset.upper()}:")
        dataset_final = final_results[final_results['dataset'] == dataset]
        
        for algo in dataset_final['algorithm'].unique():
            algo_final = dataset_final[dataset_final['algorithm'] == algo]
            mean_best = algo_final['best_value'].mean()
            std_best = algo_final['best_value'].std()
            mean_regret = algo_final['regret'].mean()
            print(f"  {algo:15s}: best = {mean_best:.4f} ± {std_best:.4f}, regret = {mean_regret:.4f}")

def print_dataset_info(dataset_name: str, A, X, y) -> None:
    """Print dataset information."""
    print(f"Dataset info:")
    print(f"  Nodes: {len(X):,}")
    print(f"  Edges: {A.nnz//2:,}")
    print(f"  Density: {A.nnz/(A.shape[0]*A.shape[1]):.6f}")
    print(f"  Target range: {y.min():.2f} to {y.max():.2f}")

def print_config(config: Dict[str, Any]) -> None:
    """Print experiment configuration."""
    print(f"📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")