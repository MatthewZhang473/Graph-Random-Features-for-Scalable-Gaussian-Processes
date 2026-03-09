import os
import pickle
import numpy as np
import networkx as nx
from datetime import datetime
import sys
import scipy.sparse as sp

# Add the correct path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from efficient_graph_gp_sparse.preprocessor import GraphPreprocessor
from efficient_graph_gp_sparse.utils_sparse import SparseLinearOperator

import numpy as np
import scipy.sparse as sp
import networkx as nx  # only used for small graphs

def generate_grid_data(n_nodes, beta_sample=1.0, noise_std=0.1, seed=42):
    """
    Central-maximum synthetic surface on a sqrt(n_nodes) x sqrt(n_nodes) grid.

    API compatibility with your previous function:
      returns dict with keys: 'A_sparse', 'G', 'y_true', 'X', 'Y'

    Notes:
      - For very large graphs, 'G' is set to None to avoid NetworkX overhead.
      - Requires n_nodes to be a perfect square.
    """
    # ---- grid shape ----
    s = int(np.sqrt(n_nodes))
    if s * s != n_nodes:
        raise ValueError("n_nodes must be a perfect square (got {}).".format(n_nodes))
    ny = nx_ = s  # (ny, nx)

    rng = np.random.default_rng(seed)

    # ---- coordinates in [0,1]^2 ----
    x = np.linspace(0, 1, nx_, dtype=np.float64)
    y = np.linspace(0, 1, ny,  dtype=np.float64)
    Xg, Yg = np.meshgrid(x, y)  # shape (ny, nx)

    # ---- smooth base + central bump (global maximum at center) ----
    Z_base = 1.2 * np.sin(2*np.pi*Xg) + 0.6 * np.cos(3*np.pi*Yg)

    cx, cy = 0.5, 0.5        # center of the peak
    lsx, lsy = 0.06, 0.06    # widths (symmetric)
    bump = 3 * np.exp(-0.5 * (((Xg - cx)/lsx)**2 + ((Yg - cy)/lsy)**2))

    Z = beta_sample * (Z_base + bump)         # true field
    y_true = Z.reshape(-1)
    y_observed = y_true + rng.normal(0.0, noise_std, size=y_true.shape)

    # ---- sparse 4-neighbour adjacency via Kronecker products (CSR) ----
    ex = np.ones(nx_)
    ey = np.ones(ny)
    Tx = sp.diags([ex[:-1], ex[:-1]], offsets=[-1, 1], shape=(nx_, nx_), format="csr")
    Ty = sp.diags([ey[:-1], ey[:-1]], offsets=[-1, 1], shape=(ny, ny),  format="csr")
    A_sparse = sp.kron(sp.eye(ny, format="csr"), Tx, format="csr") + sp.kron(Ty, sp.eye(nx_, format="csr"), format="csr")

    # ---- optional NetworkX graph for small cases; None for large ----
    # (keeps the 'G' key for API compatibility)
    if n_nodes <= 40000:  # ~200x200; adjust if you like
        G = nx.grid_2d_graph(ny, nx_)
    else:
        G = None

    return {
        'A_sparse': A_sparse,                         # scipy.sparse CSR
        'G': G,                                       # networkx.Graph or None
        'y_true': y_true,                             # shape (N,)
        'X': np.arange(n_nodes).reshape(-1, 1).astype(np.float64),
        'Y': y_observed.reshape(-1, 1),               # shape (N,1)
    }

def generate_periodic_grid_data(n_nodes, beta_sample=1.0, noise_std=0.1, seed=42):
    """
    Generate data on a periodic grid (torus topology) with smooth periodic function.
    """
    s = int(np.sqrt(n_nodes))
    if s * s != n_nodes:
        raise ValueError("n_nodes must be a perfect square (got {}).".format(n_nodes))
    
    rng = np.random.default_rng(seed)
    
    # Create periodic grid adjacency matrix
    # Each node connects to 4 neighbors with wraparound (torus topology)
    rows, cols = [], []
    
    for i in range(s):
        for j in range(s):
            node = i * s + j
            
            # Right neighbor (with wraparound)
            right = i * s + ((j + 1) % s)
            rows.extend([node, right])
            cols.extend([right, node])
            
            # Down neighbor (with wraparound)  
            down = ((i + 1) % s) * s + j
            rows.extend([node, down])
            cols.extend([down, node])
    
    # Create sparse adjacency matrix
    data = np.ones(len(rows))
    A_sparse = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    A_sparse = A_sparse.tocsr()
    
    # Generate periodic smooth function
    x = np.linspace(0, 2*np.pi, s, endpoint=False)
    y = np.linspace(0, 2*np.pi, s, endpoint=False)
    Xg, Yg = np.meshgrid(x, y)
    
    # Periodic function with multiple modes
    Z = beta_sample * (
        np.sin(Xg) * np.cos(Yg) + 
        0.5 * np.sin(2*Xg) * np.sin(2*Yg) + 
        0.3 * np.cos(3*Xg + Yg)
    )
    
    y_true = Z.reshape(-1)
    y_observed = y_true + rng.normal(0, noise_std, n_nodes)
    
    return {
        'A_sparse': A_sparse,
        'G': None,  # Skip NetworkX for scalability
        'y_true': y_true,
        'X': np.arange(n_nodes).reshape(-1, 1).astype(np.float64),
        'Y': y_observed.reshape(-1, 1),
    }

def generate_staircase_grid_data(n_nodes, beta_sample=1.0, noise_std=0.1, seed=42, n_levels=5):
    """
    Generate data on regular grid with staircase/plateau function.
    """
    s = int(np.sqrt(n_nodes))
    if s * s != n_nodes:
        raise ValueError("n_nodes must be a perfect square (got {}).".format(n_nodes))
    
    rng = np.random.default_rng(seed)
    
    # Create regular grid adjacency matrix (same as grid_data)
    ex = np.ones(s)
    ey = np.ones(s)
    Tx = sp.diags([ex[:-1], ex[:-1]], offsets=[-1, 1], shape=(s, s), format="csr")
    Ty = sp.diags([ey[:-1], ey[:-1]], offsets=[-1, 1], shape=(s, s), format="csr")
    A_sparse = sp.kron(sp.eye(s, format="csr"), Tx, format="csr") + sp.kron(Ty, sp.eye(s, format="csr"), format="csr")
    
    # Coordinates in [0,1]^2
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    Xg, Yg = np.meshgrid(x, y)
    
    # Create staircase function with plateaus
    Z = np.zeros_like(Xg)
    
    # Add diagonal staircase pattern
    diagonal_dist = Xg + Yg  # Distance along main diagonal
    level_width = 2.0 / n_levels
    
    for level in range(n_levels):
        level_start = level * level_width
        level_end = (level + 1) * level_width
        
        # Create mask for this level
        mask = (diagonal_dist >= level_start) & (diagonal_dist < level_end)
        
        # Assign level value with some noise for smoothness
        level_value = level + rng.uniform(-0.2, 0.2)
        Z[mask] = level_value
    
    # Add smooth background variation
    Z_smooth = 0.3 * np.sin(2*np.pi*Xg) * np.cos(2*np.pi*Yg)
    Z = beta_sample * (Z + Z_smooth)
    
    y_true = Z.reshape(-1)
    y_observed = y_true + rng.normal(0, noise_std, n_nodes)
    
    return {
        'A_sparse': A_sparse,
        'G': nx.grid_2d_graph(s, s) if n_nodes <= 40000 else None,
        'y_true': y_true,
        'X': np.arange(n_nodes).reshape(-1, 1).astype(np.float64),
        'Y': y_observed.reshape(-1, 1),
        'metadata': {'n_levels': n_levels}
    }

def generate_circle_graph_data(n_nodes, beta_sample=1.0, noise_std=0.1, seed=42):
    """
    Generate data on a circle graph with sinusoidal function.
    """
    rng = np.random.default_rng(seed)
    
    # Create cycle graph
    G = nx.cycle_graph(n_nodes)
    A_sparse = nx.adjacency_matrix(G).tocsr()
    
    # Generate smooth sinusoidal function on circle
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    y_true = beta_sample * (2*np.sin(2*angles) + 0.5*np.cos(4*angles) + 0.3*np.sin(angles))
    y_observed = y_true + rng.normal(0, noise_std, n_nodes)
    
    return {
        'A_sparse': A_sparse,
        'G': G if n_nodes <= 40000 else None,
        'y_true': y_true,
        'X': np.arange(n_nodes).reshape(-1, 1).astype(np.float64),
        'Y': y_observed.reshape(-1, 1),
    }

def generate_grid_multimodal_data(n_nodes, beta_sample=1.0, noise_std=0.1, seed=42, n_peaks=5):
    """
    Generate multimodal data on grid graph with multiple peaks.
    """
    s = int(np.sqrt(n_nodes))
    if s * s != n_nodes:
        raise ValueError("n_nodes must be a perfect square (got {}).".format(n_nodes))
    
    rng = np.random.default_rng(seed)
    
    # Create grid adjacency matrix (same as original)
    ex = np.ones(s)
    ey = np.ones(s)
    Tx = sp.diags([ex[:-1], ex[:-1]], offsets=[-1, 1], shape=(s, s), format="csr")
    Ty = sp.diags([ey[:-1], ey[:-1]], offsets=[-1, 1], shape=(s, s), format="csr")
    A_sparse = sp.kron(sp.eye(s, format="csr"), Tx, format="csr") + sp.kron(Ty, sp.eye(s, format="csr"), format="csr")
    
    # Coordinates in [0,1]^2
    x = np.linspace(0, 1, s, dtype=np.float64)
    y = np.linspace(0, 1, s, dtype=np.float64)
    Xg, Yg = np.meshgrid(x, y)
    
    # Base smooth function
    Z_base = 0.5 * np.sin(2*np.pi*Xg) + 0.3 * np.cos(3*np.pi*Yg)
    
    # Add multiple Gaussian peaks at random locations
    Z_peaks = np.zeros_like(Xg)
    peak_locations = []
    
    for i in range(n_peaks):
        # Random peak location
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        peak_locations.append((cx, cy))
        
        # Random peak parameters
        amplitude = rng.uniform(1.0, 3.0)
        width_x = rng.uniform(0.05, 0.15)
        width_y = rng.uniform(0.05, 0.15)
        
        # Add Gaussian peak
        peak = amplitude * np.exp(-0.5 * (((Xg - cx)/width_x)**2 + ((Yg - cy)/width_y)**2))
        Z_peaks += peak
    
    Z = beta_sample * (Z_base + Z_peaks)
    y_true = Z.reshape(-1)
    y_observed = y_true + rng.normal(0, noise_std, n_nodes)
    
    return {
        'A_sparse': A_sparse,
        'G': nx.grid_2d_graph(s, s) if n_nodes <= 40000 else None,
        'y_true': y_true,
        'X': np.arange(n_nodes).reshape(-1, 1).astype(np.float64),
        'Y': y_observed.reshape(-1, 1),
        'metadata': {'peak_locations': peak_locations, 'n_peaks': n_peaks}
    }

def get_cached_data(config):
    # Include graph type in filename
    filename = f"{config.GRAPH_TYPE}_n{config.N_NODES}_beta{config.DATA_PARAMS['beta_sample']}_noise{config.DATA_PARAMS['noise_std']}_seed{config.DATA_SEED}.pkl"
    filepath = os.path.join(config.DATA_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"📁 Loading cached data: {filename}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    print(f"🎲 Generating {config.GRAPH_TYPE} data...")
    
    # Select generation function based on graph type - UPDATED
    data_generators = {
        'grid': generate_grid_data,
        'periodic_grid': generate_periodic_grid_data,
        'staircase_grid': generate_staircase_grid_data,
        'circle': generate_circle_graph_data,
        'grid_multimodal': generate_grid_multimodal_data
    }
    
    if config.GRAPH_TYPE not in data_generators:
        raise ValueError(f"Unknown graph type: {config.GRAPH_TYPE}. Choose from {list(data_generators.keys())}")
    
    # Remove kernel_std from params since generation functions don't use it
    data_params = {k: v for k, v in config.DATA_PARAMS.items() if k != 'kernel_std'}
    
    # Add graph-specific parameters - UPDATED
    if config.GRAPH_TYPE == 'staircase_grid':
        data_params['n_levels'] = getattr(config, 'N_LEVELS', 5)
    elif config.GRAPH_TYPE == 'grid_multimodal':
        data_params['n_peaks'] = getattr(config, 'N_PEAKS', 5)
    
    data = data_generators[config.GRAPH_TYPE](config.N_NODES, seed=config.DATA_SEED, **data_params)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Data saved: {filename}")
    return data

def get_step_matrices(data, config):
    # Include graph type in filename
    filename = f"step_matrices_{config.GRAPH_TYPE}_n{config.N_NODES}_seed{config.DATA_SEED}_w{config.WALKS_PER_NODE}_p{config.P_HALT}_l{config.MAX_WALK_LENGTH}.pkl"
    filepath = os.path.join(config.STEP_MATRICES_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"📁 Loading cached step matrices: {filename}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)['step_matrices_torch']
    
    print(f"🔄 Computing step matrices for {config.GRAPH_TYPE}...")
    pp = GraphPreprocessor(
        adjacency_matrix=data['A_sparse'],
        walks_per_node=config.WALKS_PER_NODE,
        p_halt=config.P_HALT,
        max_walk_length=config.MAX_WALK_LENGTH,
        random_walk_seed=config.DATA_SEED,
        load_from_disk=False,
        use_tqdm=True,
        n_processes=16
    )
    
    pp.preprocess_graph(save_to_disk=False)
    step_matrices = pp.step_matrices_scipy
    
    save_data = {
        'step_matrices_torch': step_matrices, 
        'metadata': {
            'n_nodes': config.N_NODES, 
            'graph_type': config.GRAPH_TYPE,
            'timestamp': datetime.now().isoformat()
        }
    }
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"✅ Step matrices saved: {filename}")
    
    return step_matrices

def convert_to_device(step_matrices, device):
    result = []
    for mat in step_matrices:
        tensor = GraphPreprocessor.from_scipy_csr(mat).to(device)
        result.append(SparseLinearOperator(tensor))
    return result