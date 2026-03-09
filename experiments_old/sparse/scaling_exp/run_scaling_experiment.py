#!/usr/bin/env python3
"""
Scaling Experiment Script for Dense vs Sparse GP Implementation

This script runs comprehensive scaling experiments comparing dense (GPflow) vs sparse (GPyTorch) 
implementations of Gaussian Processes on graphs across multiple graph sizes and random seeds.

Usage:
    python run_scaling_experiment.py [--data-only] [--rw-only] [--sparse-only] [--dense-only] [--config CONFIG]
"""

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import time
import psutil
import os
import sys
import gc
from tqdm import tqdm
import pickle
import json
from datetime import datetime

# GP framework imports
import torch
import gpytorch
from gpytorch import settings as gsettings
from gpytorch.kernels import MultiDeviceKernel
from linear_operator import settings
from linear_operator.utils import linear_cg
from linear_operator.operators import IdentityLinearOperator
import gpflow
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Custom imports
from efficient_graph_gp.random_walk_samplers.sampler import RandomWalk as DenseRandomWalk, Graph as DenseGraph
from efficient_graph_gp_sparse.preprocessor import GraphPreprocessor
from efficient_graph_gp.gpflow_kernels import GraphGeneralFastGRFKernel
from efficient_graph_gp_sparse.gptorch_kernels_sparse.sparse_grf_kernel import SparseGRFKernel
from efficient_graph_gp_sparse.utils_sparse import SparseLinearOperator

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Experimental configuration parameters"""
    
    def __init__(self, config_file=None):
        # Default configuration
        self.WALKS_PER_NODE = 100
        self.P_HALT = 0.1
        self.MAX_WALK_LENGTH = 3
        
        # Graph sizes
        # self.GP_GRAPH_SIZES = [2**i for i in range(5, 8)]  # 32 to 1024 nodes
        # self.GP_SPARSE_ONLY_SIZES = [2**i for i in range(8, 9)]  # 2048 to 1048576 nodes
        self.GP_GRAPH_SIZES = [2**i for i in range(5, 11)]  # 32 to 1024 nodes
        self.GP_SPARSE_ONLY_SIZES = [2**i for i in range(11, 21)]  # 2048 to 1048576 nodes
        
        # Training parameters
        self.N_EPOCHS = 50
        self.TRAIN_RATIO = 0.6
        self.NOISE_STD = 0.1
        self.INITIAL_NOISE_VARIANCE = 0.1
        self.LEARNING_RATE = 0.1
        
        # Random seeds
        self.N_REPEATS = 5
        self.RW_SEEDS = [42 + i for i in range(self.N_REPEATS)]
        
        # Data synthesis parameters
        self.DATA_SYNTHESIS_PARAMS = {
            'beta_sample': 1.0,
            'kernel_std': 1.0,
            'noise_std': 0.1,
            'splits': [0.6, 0.2, 0.2],
            'seed': 42
        }
        
        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        
        # Setup directories
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'synthetic_data')
        self.STEP_MATRICES_DIR = os.path.join(self.BASE_DIR, 'step_matrices')
        self.STATS_DIR = os.path.join(self.BASE_DIR, 'stats')
        
        # Create directories
        for dir_path in [self.DATA_DIR, self.STEP_MATRICES_DIR, self.STATS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self, filepath):
        """Save current configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def setup_frameworks(config):
    """Setup GP frameworks and device configuration"""
    # GPyTorch & Linear Operator settings
    settings.verbose_linalg._default = False
    settings._fast_covar_root_decomposition._default = False
    gsettings.max_cholesky_size._global_value = 0
    gsettings.cg_tolerance._global_value = 1e-2
    gsettings.max_lanczos_quadrature_iterations._global_value = 1
    gsettings.num_trace_samples._global_value = 64
    gsettings.min_preconditioning_size._global_value = 1e10
    
    # Device configuration
    output_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_devices = torch.cuda.device_count()
    
    # Set seeds
    torch.manual_seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    
    print(f"🚀 Framework setup complete")
    print(f"   PyTorch device: {output_device}")
    print(f"   Number of GPUs: {n_devices}")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPyTorch version: {gpytorch.__version__}")
    
    return output_device, n_devices

def generate_ring_graph_data(n_nodes, beta_sample=1.0, kernel_std=1.0, noise_std=0.1, 
                           splits=[0.6, 0.2, 0.2], seed=42, include_dense=True):
    """Generate synthetic data on a ring graph"""
    np.random.seed(seed)
    
    # Create ring graph
    G = nx.cycle_graph(n_nodes)
    A = nx.adjacency_matrix(G).tocsr()
    
    # Generate smooth function on ring
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    y_true = beta_sample * (2*np.sin(2*angles) + 0.5*np.cos(4*angles) + 0.3*np.sin(angles))
    y_observed = y_true + np.random.normal(0, noise_std, n_nodes)
    
    # Create splits
    indices = np.arange(n_nodes)
    train_size = int(splits[0] * n_nodes)
    val_size = int(splits[1] * n_nodes)
    
    train_idx = np.random.choice(indices, train_size, replace=False)
    remaining = np.setdiff1d(indices, train_idx)
    val_idx = np.random.choice(remaining, val_size, replace=False)
    test_idx = np.setdiff1d(remaining, val_idx)
    
    data_dict = {
        'A_sparse': A,
        'G': G,
        'y_true': y_true,
        'y_observed': y_observed,
        'X_train': train_idx.reshape(-1, 1).astype(np.float64),
        'y_train': y_observed[train_idx].reshape(-1, 1),
        'X_val': val_idx.reshape(-1, 1).astype(np.float64),
        'y_val': y_observed[val_idx].reshape(-1, 1),
        'X_test': test_idx.reshape(-1, 1).astype(np.float64),
        'y_test': y_observed[test_idx].reshape(-1, 1),
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    
    # Only include dense matrix for feasible sizes
    if include_dense:
        data_dict['A_dense'] = A.toarray().astype(np.float64)
    
    return data_dict

def save_data(data, filepath):
    """Save data to pickle file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_data(filepath):
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_data_filepath(n_nodes, data_dir, params, include_dense=True):
    """Generate filepath for cached data"""
    density_suffix = "_dense" if include_dense else "_sparse"
    filename = f"ring_n{n_nodes}_beta{params['beta_sample']}_std{params['kernel_std']}_noise{params['noise_std']}_seed{params['seed']}{density_suffix}.pkl"
    return os.path.join(data_dir, filename)

def generate_and_cache_data(n_nodes, data_dir, beta_sample=1.0, kernel_std=1.0, 
                          noise_std=0.1, splits=[0.6, 0.2, 0.2], seed=42, include_dense=True):
    """Generate data and cache to disk, or load from cache if exists"""
    params = {
        'beta_sample': beta_sample,
        'kernel_std': kernel_std, 
        'noise_std': noise_std,
        'seed': seed
    }
    
    filepath = get_data_filepath(n_nodes, data_dir, params, include_dense)
    
    if os.path.exists(filepath):
        return load_data(filepath)
    else:
        data = generate_ring_graph_data(n_nodes, beta_sample, kernel_std, noise_std, splits, seed, include_dense)
        save_data(data, filepath)
        return data

def save_experiment_results(df, experiment_name, stats_dir, config_params=None):
    """Save experiment results with timestamped files and configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(stats_dir, exist_ok=True)
    
    # Save main results
    main_file = os.path.join(stats_dir, f'{experiment_name}_stats.csv')
    timestamped_file = os.path.join(stats_dir, f'{experiment_name}_stats_{timestamp}.csv')
    
    df.to_csv(main_file, index=False)
    df.to_csv(timestamped_file, index=False)
    
    # Save configuration if provided
    if config_params:
        config_summary = {
            'timestamp': timestamp,
            'total_experiments': len(df),
            'experiment_name': experiment_name,
            **config_params
        }
        
        with open(os.path.join(stats_dir, f'{experiment_name}_config_{timestamp}.json'), 'w') as f:
            json.dump(config_summary, f, indent=2)
    
    # Compute and save summary statistics
    if len(df) > 0:
        if 'n_nodes' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            summary = df.groupby('n_nodes')[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(4)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            summary = df[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(4)
        
        summary.to_csv(os.path.join(stats_dir, f'{experiment_name}_summary_{timestamp}.csv'))
    
    print(f"📁 {experiment_name} results saved:")
    print(f"   Main file: {main_file}")
    print(f"   Timestamped: {timestamped_file}")
    
    return timestamped_file

# =============================================================================
# DATA SYNTHESIS
# =============================================================================

def synthesize_all_data(config):
    """Generate synthetic data for all graph sizes"""
    print(f"🎲 Synthesizing ring graph data for all graph sizes...")
    
    # Separate feasible sizes (can compute dense) from sparse-only sizes
    feasible_sizes = config.GP_GRAPH_SIZES  # These can have dense matrices
    sparse_only_sizes = config.GP_SPARSE_ONLY_SIZES  # These are sparse-only
    
    print(f"   Feasible sizes (with dense): {feasible_sizes}")
    print(f"   Sparse-only sizes: {sparse_only_sizes}")
    
    # Generate data for feasible sizes (include dense matrices)
    for n_nodes in tqdm(feasible_sizes, desc="Generating feasible datasets (dense+sparse)"):
        data = generate_and_cache_data(
            n_nodes=n_nodes,
            data_dir=config.DATA_DIR,
            include_dense=True,  # Include dense matrix
            **config.DATA_SYNTHESIS_PARAMS
        )
        print(f"   ✓ Generated data for {n_nodes} nodes (dense: {data.get('A_dense') is not None})")
        
        # Clean up memory for larger datasets
        del data
        if n_nodes >= 1000:
            gc.collect()
    
    # Generate data for sparse-only sizes (exclude dense matrices)
    for n_nodes in tqdm(sparse_only_sizes, desc="Generating sparse-only datasets"):
        data = generate_and_cache_data(
            n_nodes=n_nodes,
            data_dir=config.DATA_DIR,
            include_dense=False,  # Do NOT include dense matrix
            **config.DATA_SYNTHESIS_PARAMS
        )
        print(f"   ✓ Generated data for {n_nodes} nodes (sparse-only)")
        
        # Clean up memory for large datasets
        del data
        gc.collect()
    
    print(f"✅ Data synthesis complete. Files stored in: {config.DATA_DIR}")
    return True

# =============================================================================
# RANDOM WALK SAMPLING
# =============================================================================

def run_sparse_rw_sampling(data, config, rw_seed, n_nodes):
    """Run sparse random walk sampling for a single graph"""
    start_time = time.time()
    pp_sparse = GraphPreprocessor(
        adjacency_matrix=data['A_sparse'],
        walks_per_node=config.WALKS_PER_NODE,
        p_halt=config.P_HALT,
        max_walk_length=config.MAX_WALK_LENGTH,
        random_walk_seed=rw_seed,
        load_from_disk=False,
        use_tqdm=False,
        n_processes=4
    )
    
    pp_sparse.preprocess_graph(save_to_disk=False)
    step_matrices_scipy = pp_sparse.step_matrices_scipy
    sparse_rw_time = time.time() - start_time
    
    # Calculate object sizes using scipy matrices
    sparse_total_nnz = sum(mat.nnz for mat in step_matrices_scipy)
    sparse_size_mb = sparse_total_nnz * 16 / (1024**2)
    sparse_dense_equiv_mb = sum(mat.shape[0] * mat.shape[1] * 8 for mat in step_matrices_scipy) / (1024**2)
    
    return {
        'time': sparse_rw_time,
        'step_matrices': step_matrices_scipy,
        'total_nnz': sparse_total_nnz,
        'size_mb': sparse_size_mb,
        'dense_equiv_mb': sparse_dense_equiv_mb,
        'avg_nnz_per_matrix': sparse_total_nnz / len(step_matrices_scipy),
        'sparsity': sparse_total_nnz / sum(mat.shape[0] * mat.shape[1] for mat in step_matrices_scipy)
    }

def run_dense_rw_sampling(data, config, rw_seed, n_nodes):
    """Run dense random walk sampling for a single graph"""
    start_time = time.time()
    dense_graph = DenseGraph(data['A_dense'])
    dense_sampler = DenseRandomWalk(dense_graph, seed=rw_seed)
    
    dense_step_matrices = dense_sampler.get_random_walk_matrices(
        config.WALKS_PER_NODE, config.P_HALT, config.MAX_WALK_LENGTH
    )
    dense_rw_time = time.time() - start_time
    
    dense_size_mb = dense_step_matrices.nbytes / (1024**2)
    
    return {
        'time': dense_rw_time,
        'step_matrices': dense_step_matrices,
        'size_mb': dense_size_mb
    }

def save_step_matrices(step_matrices_dir, method, n_nodes, rw_seed, step_matrices, config_dict):
    """Save step matrices to disk and return file size"""
    filename = f"step_matrices_{method}_n{n_nodes}_seed{rw_seed}.pkl"
    filepath = os.path.join(step_matrices_dir, filename)
    
    save_data = {
        'step_matrices_torch' if method == 'sparse' else 'step_matrices': step_matrices,
        'n_nodes': n_nodes,
        'seed': rw_seed,
        'method': method,
        'config': config_dict
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    return os.path.getsize(filepath) / (1024**2)

def run_rw_sampling_experiment(config):
    """Run random walk sampling experiments for all graph sizes and seeds"""
    print(f"🚶 Running RW sampling experiments...")
    
    all_sizes = config.GP_GRAPH_SIZES + config.GP_SPARSE_ONLY_SIZES
    all_results = []
    
    config_dict = {'walks_per_node': config.WALKS_PER_NODE, 'p_halt': config.P_HALT, 'max_walk_length': config.MAX_WALK_LENGTH}
    
    for n_nodes in tqdm(all_sizes, desc="Graph sizes"):
        print(f"\nProcessing {n_nodes} nodes...")
        
        # Determine if this is a feasible size or sparse-only
        is_feasible = n_nodes in config.GP_GRAPH_SIZES
        include_dense = is_feasible
        
        # Load synthetic data with appropriate density setting
        data = generate_and_cache_data(
            n_nodes=n_nodes, 
            data_dir=config.DATA_DIR, 
            include_dense=include_dense,
            **config.DATA_SYNTHESIS_PARAMS
        )
        
        # Determine if we should run dense (only for feasible sizes with reasonable memory requirements)
        run_dense = is_feasible and n_nodes <= 1024
        
        for seed_idx, rw_seed in enumerate(config.RW_SEEDS):
            print(f"  Seed {seed_idx + 1}/{len(config.RW_SEEDS)} (seed={rw_seed})")
            
            # Run sparse sampling (always)
            sparse_result = run_sparse_rw_sampling(data, config, rw_seed, n_nodes)
            sparse_file_size = save_step_matrices(config.STEP_MATRICES_DIR, 'sparse', n_nodes, rw_seed, 
                                                sparse_result['step_matrices'], config_dict)
            
            # Run dense sampling only if applicable
            if run_dense:
                if 'A_dense' not in data:
                    print(f"    ⚠️  Dense matrix not available for {n_nodes} nodes, skipping dense sampling")
                    dense_result = None
                    dense_file_size = None
                else:
                    dense_result = run_dense_rw_sampling(data, config, rw_seed, n_nodes)
                    dense_file_size = save_step_matrices(config.STEP_MATRICES_DIR, 'dense', n_nodes, rw_seed,
                                                       dense_result['step_matrices'], config_dict)
            else:
                dense_result = None
                dense_file_size = None
            
            # Compile statistics
            stat_entry = {
                'n_nodes': n_nodes,
                'n_edges': data['A_sparse'].nnz // 2,
                'seed': rw_seed,
                'sparse_rw_time': sparse_result['time'],
                'dense_rw_time': dense_result['time'] if dense_result else None,
                'sparse_size_mb': sparse_result['size_mb'],
                'dense_size_mb': dense_result['size_mb'] if dense_result else None,
                'sparse_dense_equiv_mb': sparse_result['dense_equiv_mb'],
                'compression_ratio': (dense_result['size_mb'] / sparse_result['size_mb'] 
                                    if dense_result and sparse_result['size_mb'] > 0 else np.nan),
                'time_speedup': (dense_result['time'] / sparse_result['time'] 
                               if dense_result else np.nan),
                'sparse_file_size_mb': sparse_file_size,
                'dense_file_size_mb': dense_file_size,
                'sparse_total_nnz': sparse_result['total_nnz'],
                'sparse_avg_nnz_per_matrix': sparse_result['avg_nnz_per_matrix'],
                'graph_sparsity': data['A_sparse'].nnz / (n_nodes**2),
                'step_matrix_sparsity': sparse_result['sparsity'],
                'run_dense': run_dense,
                'is_feasible': is_feasible,
                'has_dense_data': 'A_dense' in data
            }
            all_results.append(stat_entry)
            
            # Cleanup
            del sparse_result
            if dense_result:
                del dense_result
            if n_nodes >= 1000:
                gc.collect()
        
        del data
        if n_nodes >= 1000:
            gc.collect()
    
    # Save and summarize results
    rw_df = pd.DataFrame(all_results)
    save_experiment_results(rw_df, 'rw_sampling', config.STATS_DIR)
    
    print(f"\n✅ RW sampling complete! Processed {len(all_results)} experiments")
    return rw_df

# =============================================================================
# SPARSE GP MODEL
# =============================================================================

class SparseGraphGPModel(gpytorch.models.ExactGP):
    """Sparse Graph GP Model with pathwise conditioning prediction"""
    
    def __init__(self, x_train, y_train, likelihood, step_matrices_torch, max_walk_length):
        super().__init__(x_train, y_train, likelihood)
        self.x_train = x_train
        self.y_train = y_train
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = SparseGRFKernel(
            max_walk_length=max_walk_length, 
            step_matrices_torch=step_matrices_torch
        )
        self.num_nodes = step_matrices_torch[0].shape[0]
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def predict(self, x_test, n_samples=64):
        """Batch pathwise conditioning prediction"""
        num_train = self.x_train.shape[0]
        train_indices = self.x_train.int().flatten()
        test_indices = x_test.int().flatten()
        
        # Feature matrices
        phi = self.covar_module._get_feature_matrix()
        phi_train = phi[train_indices, :]
        phi_test = phi[test_indices, :]
        
        # Covariance matrices
        K_train_train = phi_train @ phi_train.T
        K_test_train = phi_test @ phi_train.T
        
        # Noise setup
        noise_variance = self.likelihood.noise.item()
        noise_std = torch.sqrt(torch.tensor(noise_variance, device=x_test.device))
        A = K_train_train + noise_variance * IdentityLinearOperator(num_train, device=x_test.device)
        
        # Batch samples
        eps1_batch = torch.randn(n_samples, self.num_nodes, device=x_test.device)
        eps2_batch = noise_std * torch.randn(n_samples, num_train, device=x_test.device)
        
        # Prior samples
        f_test_prior_batch = eps1_batch @ phi_test.T
        f_train_prior_batch = eps1_batch @ phi_train.T
        
        # CG solve
        b_batch = self.y_train.unsqueeze(0) - (f_train_prior_batch + eps2_batch)
        v_batch = linear_cg(A._matmul, b_batch.T, tolerance=gsettings.cg_tolerance.value())
        
        # Posterior
        return f_test_prior_batch + (K_test_train @ v_batch).T

def load_step_matrices_from_file(filepath, device):
    """Load and convert step matrices to torch tensors"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    step_matrices_scipy = data['step_matrices_torch']
    
    step_matrices_torch = []
    for mat in step_matrices_scipy:
        tensor = GraphPreprocessor.from_scipy_csr(mat).to(device)
        step_matrices_torch.append(SparseLinearOperator(tensor))
    
    return step_matrices_torch

def run_sparse_gp_experiment(config, output_device, n_nodes, rw_seed):
    """Run complete sparse GP experiment for one graph size and seed"""
    # Determine if this is a feasible size or sparse-only
    is_feasible = n_nodes in config.GP_GRAPH_SIZES
    include_dense = is_feasible
    
    # Load data with appropriate density setting
    data = generate_and_cache_data(
        n_nodes, 
        config.DATA_DIR, 
        include_dense=include_dense,
        **config.DATA_SYNTHESIS_PARAMS
    )
    
    # Convert to torch tensors on GPU
    data_torch = {
        'X_train': torch.tensor(data['train_idx'], dtype=torch.float32, device=output_device).unsqueeze(1),
        'y_train': torch.tensor(data['y_train'].flatten(), dtype=torch.float32, device=output_device),
        'X_test': torch.tensor(data['test_idx'], dtype=torch.float32, device=output_device).unsqueeze(1),
        'y_test': torch.tensor(data['y_test'].flatten(), dtype=torch.float32, device=output_device)
    }
    
    # Load step matrices
    step_matrices_file = os.path.join(config.STEP_MATRICES_DIR, f'step_matrices_sparse_n{n_nodes}_seed{rw_seed}.pkl')
    if not os.path.exists(step_matrices_file):
        return None
    
    step_matrices_torch = load_step_matrices_from_file(step_matrices_file, output_device)
    
    # Train model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = SparseGraphGPModel(
        data_torch['X_train'], data_torch['y_train'], 
        likelihood, step_matrices_torch, config.MAX_WALK_LENGTH
    ).to(output_device)
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    train_time_start = time.time()
    for i in range(config.N_EPOCHS):
        optimizer.zero_grad()
        output = model(data_torch['X_train'])
        loss = -mll(output, data_torch['y_train'])
        loss.backward()
        optimizer.step()
    train_time = time.time() - train_time_start
    
    # Evaluate model
    model.eval()
    likelihood.eval()
    
    inference_time_start = time.time()
    with torch.no_grad():
        test_samples = model.predict(data_torch['X_test'], n_samples=64)
        test_mean = test_samples.mean(dim=0)
        test_std = test_samples.std(dim=0)
    inference_time = time.time() - inference_time_start
    
    test_rmse = torch.sqrt(torch.mean((data_torch['y_test'] - test_mean) ** 2)).item()
    
    return {
        'n_nodes': n_nodes,
        'seed': rw_seed,
        'n_train': len(data['train_idx']),
        'n_test': len(data['test_idx']),
        'train_time': train_time, 
        'inference_time': inference_time,
        'total_time': train_time + inference_time,
        'test_rmse': test_rmse,
        'noise_variance': likelihood.noise.item(),
        'modulator_l2': np.linalg.norm(model.covar_module.modulator_vector.detach().cpu().numpy())
    }

def run_sparse_gp_scaling_experiment(config, output_device):
    """Run sparse GP experiments across all sizes and seeds"""
    print(f"🚀 Running sparse GP experiments on {output_device}...")
    
    gp_results = []
    all_sizes = config.GP_GRAPH_SIZES + config.GP_SPARSE_ONLY_SIZES
    
    for n_nodes in tqdm(all_sizes, desc="Graph sizes"):
        for rw_seed in config.RW_SEEDS:
            result = run_sparse_gp_experiment(config, output_device, n_nodes, rw_seed)
            if result:
                gp_results.append(result)
                print(f"  ✓ {n_nodes} nodes, seed {rw_seed}: RMSE={result['test_rmse']:.4f}")
            else:
                print(f"  ✗ {n_nodes} nodes, seed {rw_seed}: Failed")
            
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Save results
    if gp_results:
        gp_df = pd.DataFrame(gp_results)
        config_params = {
            'graph_sizes': sorted(gp_df['n_nodes'].unique().tolist()),
            'seeds': sorted(gp_df['seed'].unique().tolist()),
            'n_epochs': config.N_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'walks_per_node': config.WALKS_PER_NODE,
            'p_halt': config.P_HALT,
            'max_walk_length': config.MAX_WALK_LENGTH,
            'device': str(output_device)
        }
        
        save_experiment_results(gp_df, 'sparse_gp_scaling', config.STATS_DIR, config_params)
        print(f"✅ Sparse GP scaling complete! Processed {len(gp_results)} experiments")
    else:
        print("❌ No sparse GP experiments completed successfully")
        gp_df = pd.DataFrame()
    
    return gp_df

# =============================================================================
# DENSE GP MODEL
# =============================================================================

class DenseGPModel:
    """Dense GP model using GPflow"""
    
    def __init__(self, data, config, step_matrices_dense=None):
        self.data = data
        self.config = config
        self.step_matrices_dense = step_matrices_dense
        self.model = None
        
    def build_model(self):
        """Build GPflow model with dense kernel"""
        self.kernel = GraphGeneralFastGRFKernel(
            self.data['A_dense'], 
            walks_per_node=self.config.WALKS_PER_NODE, 
            p_halt=self.config.P_HALT, 
            max_walk_length=self.config.MAX_WALK_LENGTH,
            step_matrices=self.step_matrices_dense,
            use_tqdm=False
        )
        
        self.model = gpflow.models.GPR(
            data=(self.data['X_train'], self.data['y_train']), 
            kernel=self.kernel, 
            noise_variance=self.config.INITIAL_NOISE_VARIANCE
        )
        
    def train(self, n_epochs=50):
        """Train the model"""
        optimizer = gpflow.optimizers.Scipy()
        
        def objective():
            return -self.model.log_marginal_likelihood()
        
        optimizer.minimize(
            objective,
            self.model.trainable_variables,
            options={"maxiter": n_epochs},
            compile=False
        )
        
    def predict(self, X_test):
        """Make predictions"""
        mean_pred, var_pred = self.model.predict_f(X_test)
        return mean_pred.numpy(), var_pred.numpy()

def run_dense_gp_experiment(config, n_nodes, rw_seed):
    """Run complete dense GP experiment for one graph size and seed"""
    # Only run dense GP for feasible sizes
    if n_nodes not in config.GP_GRAPH_SIZES:
        print(f"    ⚠️  {n_nodes} nodes not in feasible sizes, skipping dense GP")
        return None
    
    # Load data with dense matrices
    data = generate_and_cache_data(
        n_nodes, 
        config.DATA_DIR, 
        include_dense=True,  # Dense GP requires dense matrix
        **config.DATA_SYNTHESIS_PARAMS
    )
    
    if 'A_dense' not in data:
        print(f"    ⚠️  Dense matrix not available for {n_nodes} nodes")
        return None
    
    # Load dense step matrices if available
    step_matrices_file = os.path.join(config.STEP_MATRICES_DIR, f'step_matrices_dense_n{n_nodes}_seed{rw_seed}.pkl')
    step_matrices_dense = None
    
    if os.path.exists(step_matrices_file):
        with open(step_matrices_file, 'rb') as f:
            step_data = pickle.load(f)
            step_matrices_dense = step_data['step_matrices']
        print(f"    Loaded pre-computed dense step matrices with shape {step_matrices_dense.shape}")
    else:
        print(f"    Computing dense step matrices on-the-fly")
    
    # Train model - pass step matrices, not adjacency matrix
    model_wrapper = DenseGPModel(data, config, step_matrices_dense)
    model_wrapper.build_model()
    
    train_time_start = time.time()
    model_wrapper.train(config.N_EPOCHS)
    train_time = time.time() - train_time_start
    
    # Evaluate model
    inference_time_start = time.time()
    test_mean, test_var = model_wrapper.predict(data['X_test'])
    inference_time = time.time() - inference_time_start
    
    test_std = np.sqrt(test_var.flatten())
    test_rmse = np.sqrt(mean_squared_error(data['y_test'], test_mean))
    
    return {
        'n_nodes': n_nodes,
        'seed': rw_seed,
        'n_train': len(data['train_idx']),
        'n_test': len(data['test_idx']),
        'train_time': train_time,
        'inference_time': inference_time,
        'total_time': train_time + inference_time,
        'test_rmse': test_rmse,
        'noise_variance': float(model_wrapper.model.likelihood.variance.numpy()),
        'modulator_l2': np.linalg.norm(model_wrapper.kernel.modulator_vector.numpy())
    }

def run_dense_gp_scaling_experiment(config):
    """Run dense GP experiments across all feasible sizes and seeds"""
    print(f"🚀 Running dense GP experiments...")
    
    # Run only on sizes that have dense step matrices
    feasible_sizes = config.GP_GRAPH_SIZES
    gp_results = []
    
    for n_nodes in tqdm(feasible_sizes, desc="Graph sizes"):
        for rw_seed in config.RW_SEEDS:
            result = run_dense_gp_experiment(config, n_nodes, rw_seed)
            if result:
                gp_results.append(result)
                print(f"  ✓ {n_nodes} nodes, seed {rw_seed}: RMSE={result['test_rmse']:.4f}")
            else:
                print(f"  ✗ {n_nodes} nodes, seed {rw_seed}: Failed")
            
            # Cleanup memory
            gc.collect()
    
    # Save results
    if gp_results:
        gp_df = pd.DataFrame(gp_results)
        config_params = {
            'graph_sizes': sorted(gp_df['n_nodes'].unique().tolist()),
            'seeds': sorted(gp_df['seed'].unique().tolist()),
            'n_epochs': config.N_EPOCHS,
            'walks_per_node': config.WALKS_PER_NODE,
            'p_halt': config.P_HALT,
            'max_walk_length': config.MAX_WALK_LENGTH,
            'framework': 'gpflow_dense'
        }
        
        save_experiment_results(gp_df, 'dense_gp_scaling', config.STATS_DIR, config_params)
        print(f"✅ Dense GP scaling complete! Processed {len(gp_results)} experiments")
    else:
        print("❌ No dense GP experiments completed successfully")
        gp_df = pd.DataFrame()
    
    return gp_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_summary_statistics(results_dict):
    """Print summary of all results"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    if 'rw_df' in results_dict and len(results_dict['rw_df']) > 0:
        rw_df = results_dict['rw_df']
        comparison_df = rw_df[rw_df['run_dense'] == True]
        sparse_only_df = rw_df[rw_df['run_dense'] == False]
        
        print(f"\n📊 Random Walk Performance:")
        if len(comparison_df) > 0:
            avg_speedup = comparison_df['time_speedup'].mean()
            avg_compression = comparison_df['compression_ratio'].mean()
            print(f"   • Average Speedup: {avg_speedup:.2f}x")
            print(f"   • Average Compression: {avg_compression:.1f}x")
        
        print(f"   • Total Experiments: {len(rw_df)}")
        print(f"   • Sparse-only (large graphs): {len(sparse_only_df)} experiments")
    
    if 'sparse_gp_df' in results_dict and len(results_dict['sparse_gp_df']) > 0:
        sparse_df = results_dict['sparse_gp_df']
        print(f"\n🚀 Sparse GP Performance:")
        print(f"   • Graph sizes: {sorted(sparse_df['n_nodes'].unique().tolist())}")
        print(f"   • Average RMSE: {sparse_df['test_rmse'].mean():.4f}")
        print(f"   • Average training time: {sparse_df['train_time'].mean():.2f}s")
        print(f"   • Total experiments: {len(sparse_df)}")
    
    if 'dense_gp_df' in results_dict and len(results_dict['dense_gp_df']) > 0:
        dense_df = results_dict['dense_gp_df']
        print(f"\n🚀 Dense GP Performance:")
        print(f"   • Graph sizes: {sorted(dense_df['n_nodes'].unique().tolist())}")
        print(f"   • Average RMSE: {dense_df['test_rmse'].mean():.4f}")
        print(f"   • Average training time: {dense_df['train_time'].mean():.2f}s")
        print(f"   • Total experiments: {len(dense_df)}")
    
    # Compare if both exist
    if ('sparse_gp_df' in results_dict and 'dense_gp_df' in results_dict and 
        len(results_dict['sparse_gp_df']) > 0 and len(results_dict['dense_gp_df']) > 0):
        
        sparse_df = results_dict['sparse_gp_df']
        dense_df = results_dict['dense_gp_df']
        
        # Find overlapping sizes
        sparse_sizes = set(sparse_df['n_nodes'].unique())
        dense_sizes = set(dense_df['n_nodes'].unique())
        common_sizes = sparse_sizes & dense_sizes
        
        if common_sizes:
            print(f"\n⚖️  Dense vs Sparse Comparison (common sizes: {sorted(common_sizes)}):")
            
            sparse_common = sparse_df[sparse_df['n_nodes'].isin(common_sizes)]
            dense_common = dense_df[dense_df['n_nodes'].isin(common_sizes)]
            
            avg_sparse_time = sparse_common['total_time'].mean()
            avg_dense_time = dense_common['total_time'].mean()
            avg_sparse_rmse = sparse_common['test_rmse'].mean()
            avg_dense_rmse = dense_common['test_rmse'].mean()
            
            if avg_sparse_time > 0:
                speedup = avg_dense_time / avg_sparse_time
                print(f"   • Speed improvement: {speedup:.2f}x")
            
            rmse_diff = abs(avg_sparse_rmse - avg_dense_rmse)
            print(f"   • RMSE difference: {rmse_diff:.4f}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run scaling experiments for Dense vs Sparse GP implementations")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--data-only', action='store_true', help='Only run data synthesis')
    parser.add_argument('--rw-only', action='store_true', help='Only run random walk sampling')
    parser.add_argument('--sparse-only', action='store_true', help='Only run sparse GP experiments')
    parser.add_argument('--dense-only', action='store_true', help='Only run dense GP experiments')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config(args.config)
    
    # Setup frameworks
    output_device, n_devices = setup_frameworks(config)
    
    # Save configuration
    config_file = os.path.join(config.STATS_DIR, f'experiment_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    config.save_config(config_file)
    print(f"📋 Configuration saved to: {config_file}")
    
    # Track results
    results = {}
    
    try:
        # 1. Data synthesis
        if not any([args.rw_only, args.sparse_only, args.dense_only]):
            print(f"\n{'='*20} DATA SYNTHESIS {'='*20}")
            synthesize_all_data(config)
        
        # 2. Random walk sampling
        if not any([args.data_only, args.sparse_only, args.dense_only]):
            print(f"\n{'='*20} RANDOM WALK SAMPLING {'='*20}")
            results['rw_df'] = run_rw_sampling_experiment(config)
        
        # 3. Sparse GP experiments
        if not any([args.data_only, args.rw_only, args.dense_only]):
            print(f"\n{'='*20} SPARSE GP EXPERIMENTS {'='*20}")
            results['sparse_gp_df'] = run_sparse_gp_scaling_experiment(config, output_device)
        
        # 4. Dense GP experiments
        if not any([args.data_only, args.rw_only, args.sparse_only]):
            print(f"\n{'='*20} DENSE GP EXPERIMENTS {'='*20}")
            results['dense_gp_df'] = run_dense_gp_scaling_experiment(config)
        
        # Print summary
        print_summary_statistics(results)
        
        print(f"\n🎉 Scaling experiment completed successfully!")
        print(f"📁 Results saved to: {config.STATS_DIR}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
