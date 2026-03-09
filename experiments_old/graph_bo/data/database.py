import os
import numpy as np
import pandas as pd
import pickle
import pandas as pd
import networkx as nx
import gzip
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Any
import warnings

class GraphDataLoader:
    """Database class for loading and caching graph datasets."""
    
    def __init__(self, data_root="raw_data", cache_dir=None):
        # Set default cache directory at the same level as data_root
        if cache_dir is None:
            cache_dir = "processed_data"
        
        # Handle relative paths - if data_root doesn't exist, try going up one level
        if not os.path.exists(data_root) and not os.path.isabs(data_root):
            parent_data_root = os.path.join("..", data_root)
            if os.path.exists(parent_data_root):
                data_root = parent_data_root
                # Also adjust cache_dir to be at the same level
                cache_dir = os.path.join("..", cache_dir)
        
        self.data_root = data_root
        self.cache_dir = cache_dir
        self._cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache directory: {os.path.abspath(cache_dir)}")
        
        # Dataset configurations
        self.dataset_configs = {
            'facebook': {
                'edges_file': 'facebook_large/musae_facebook_edges.csv',
                'targets_file': 'facebook_large/musae_facebook_target.csv',
                'loader': self._load_facebook,
                'type': 'social_networks'
            },
            'youtube': {
                'graph_file': 'com-youtube.ungraph.txt.gz',
                'loader': self._load_youtube,
                'type': 'social_networks'
            },
            'twitch': {
                'edges_file': 'large_twitch_edges.csv',
                'features_file': 'large_twitch_features.csv',
                'loader': self._load_twitch,
                'type': 'social_networks'
            },
            'enron': {
                'graph_file': 'email-Enron.txt.gz',
                'loader': self._load_enron,
                'type': 'social_networks'
            },
            '500hpa': {
                'data_file': 'wind_data_processed_500hPa.npz',
                'loader': self._load_wind_data,
                'type': 'wind_interpolation',
                'subdir': '500hPa'
            },
            '800hpa': {
                'data_file': 'wind_data_processed_800hPa.npz',
                'loader': self._load_wind_data,
                'type': 'wind_interpolation',
                'subdir': '800hPa'
            },
            '1000hpa': {
                'data_file': 'wind_data_processed_1000hPa.npz',
                'loader': self._load_wind_data,
                'type': 'wind_interpolation',
                'subdir': '1000hPa'
            },
            '500hpa_wide': {
                'data_file': 'wind_data_processed_500hPa_wide.npz',
                'loader': self._load_wind_data,
                'type': 'wind_interpolation',
                'subdir': '500hPa_wide'
            },
            '800hpa_wide': {
                'data_file': 'wind_data_processed_800hPa_wide.npz',
                'loader': self._load_wind_data,
                'type': 'wind_interpolation',
                'subdir': '800hPa_wide'
            },
            '1000hpa_wide': {
                'data_file': 'wind_data_processed_1000hPa_wide.npz',
                'loader': self._load_wind_data,
                'type': 'wind_interpolation',
                'subdir': '1000hPa_wide'
            },
            'single_modal': {
                'data_file': 'synthetic_single_modal_1000x1000.npz',
                'loader': self._load_synthetic_data,
                'type': 'synthetic',
                'subdir': 'single_modal'
            },
            'multi_modal': {
                'data_file': 'synthetic_multimodal_1000x1000.npz',
                'loader': self._load_synthetic_data,
                'type': 'synthetic',
                'subdir': 'multi-modal'
            },
            'bimodal': {
                'data_file': 'synthetic_bimodal_100x100.npz',
                'loader': self._load_synthetic_data,
                'type': 'synthetic',
                'subdir': 'bimodal'
            },
            'community': {
                'data_file': 'synthetic_community_10k.npz',
                'loader': self._load_synthetic_data,
                'type': 'synthetic',
                'subdir': 'community'
            },
            'circular': {
                'data_file': 'synthetic_circular_10k.npz',
                'loader': self._load_synthetic_data,
                'type': 'synthetic',
                'subdir': 'circular'
            }
        }

    def __call__(self, dataset_name: str, dataset_type: str = None, force_reload: bool = False) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """
        Load dataset with caching.
        
        Args:
            dataset_name: Name of the dataset
            force_reload: Force reload from source files
            
        Returns:
            For all datasets: Tuple of (adjacency_matrix, node_indices, target_values)
            - Graph datasets: target_values = node_degrees
            - Wind datasets: target_values = wind_speed_magnitudes
        """
        if dataset_name not in self.dataset_configs:
            available = list(self.dataset_configs.keys())
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
        
        # Check memory cache first
        if not force_reload and dataset_name in self._cache:
            return self._cache[dataset_name]
        
        # Check disk cache
        cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
        if not force_reload and os.path.exists(cache_path):
            print(f"Loading {dataset_name} from cache...")
            data = self._load_from_cache(cache_path)
            self._cache[dataset_name] = data
            return data
        
        # Load from source and cache
        print(f"Loading {dataset_name} from source files...")
        data = self._load_from_source(dataset_name, dataset_type)
        
        # Save to disk cache
        self._save_to_cache(data, cache_path, dataset_name)
        
        # Save to memory cache
        self._cache[dataset_name] = data
        
        return data
    
    def _load_from_source(self, dataset_name: str, dataset_type: str = None) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load dataset from source files."""
        config = self.dataset_configs[dataset_name]
        
        if dataset_type:
            dataset_path = os.path.join(self.data_root, dataset_type, dataset_name)
        else:
            # For wind datasets, use the specific subdirectory structure
            if 'subdir' in config:
                dataset_path = os.path.join(self.data_root, config['type'], config['subdir'])
            else:
                # For regular datasets, try type subdirectory first, then direct
                dataset_type = config.get('type', '')
                if dataset_type:
                    dataset_path = os.path.join(self.data_root, dataset_type, dataset_name)
                    if not os.path.exists(dataset_path):
                        dataset_path = os.path.join(self.data_root, dataset_name)
                else:
                    dataset_path = os.path.join(self.data_root, dataset_name)

        # Check if dataset directory exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        # Call the appropriate loader
        return config['loader'](dataset_path)
    
    def _load_facebook(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load Facebook dataset."""
        edges_path = os.path.join(dataset_path, "facebook_large/musae_facebook_edges.csv")
        
        if not os.path.exists(edges_path):
            raise FileNotFoundError(f"Facebook edges file not found: {edges_path}")
        
        # Load edges
        edges_df = pd.read_csv(edges_path)
        
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(edges_df, source='id_1', target='id_2')
        
        # Convert to adjacency matrix and enforce CSR format
        adjacency_matrix = nx.adjacency_matrix(G).tocsr()
        
        # Get node indices and degrees
        X = np.array(list(G.nodes()))
        y = np.array([G.degree(node) for node in X])
        
        return adjacency_matrix, X, y
    
    def _load_youtube(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load YouTube dataset."""
        graph_path = os.path.join(dataset_path, "com-youtube.ungraph.txt.gz")
        
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"YouTube graph file not found: {graph_path}")
        
        G = nx.Graph()
        
        with gzip.open(graph_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                u, v = map(int, line.strip().split())
                G.add_edge(u, v)
        
        # Convert to adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).tocsr()
        
        # Get node indices and degrees
        X = np.array(list(G.nodes()))
        y = np.array([G.degree(node) for node in X])
        
        return adjacency_matrix, X, y
    
    def _load_twitch(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load Twitch dataset."""
        edges_path = os.path.join(dataset_path, "large_twitch_edges.csv")
        
        if not os.path.exists(edges_path):
            raise FileNotFoundError(f"Twitch edges file not found: {edges_path}")
        
        # Load edges
        edges_df = pd.read_csv(edges_path)
        
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(edges_df, source='numeric_id_1', target='numeric_id_2')
        
        # Convert to adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).tocsr()
        
        # Get node indices and degrees
        X = np.array(list(G.nodes()))
        y = np.array([G.degree(node) for node in X])
        
        return adjacency_matrix, X, y
    
    def _load_enron(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load Enron dataset."""
        graph_path = os.path.join(dataset_path, "email-Enron.txt.gz")
        
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Enron graph file not found: {graph_path}")
        
        G = nx.Graph()
        
        with gzip.open(graph_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                u, v = map(int, line.strip().split())
                G.add_edge(u, v)
        
        # Convert to adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).tocsr()
        
        # Get node indices and degrees
        X = np.array(list(G.nodes()))
        y = np.array([G.degree(node) for node in X])
        
        return adjacency_matrix, X, y
    
    def _load_wind_data(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load wind interpolation dataset."""
        # Find the dataset name by looking up which config has this path
        dataset_name = None
        for name, config in self.dataset_configs.items():
            if config.get('loader') == self._load_wind_data:
                expected_path = os.path.join(self.data_root, config['type'], config['subdir'])
                if expected_path == dataset_path:
                    dataset_name = name
                    break
        
        if dataset_name is None:
            # Fallback: extract from path
            dataset_name = os.path.basename(dataset_path).lower() + 'hpa'
        
        config = self.dataset_configs[dataset_name]
        
        # The exact path structure: wind_interpolation/500hPa/wind_data_processed_500hPa.npz
        data_file_path = os.path.join(dataset_path, config['data_file'])
        
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Wind data file not found: {data_file_path}")
        
        # Load the processed wind data
        data = np.load(data_file_path)
        
        # Reconstruct sparse adjacency matrix
        A_data = data['A_data']
        A_indices = data['A_indices']
        A_indptr = data['A_indptr']
        A_shape = data['A_shape']
        adjacency_matrix = csr_matrix((A_data, A_indices, A_indptr), shape=A_shape)
        
        # Load node indices (consistent with social network datasets)
        X = data['X']  # shape (num_nodes,) - node indices [0, 1, 2, ...]
        
        # Load wind speed magnitudes (consistent target format)
        y = data['y']  # shape (num_nodes,) - wind speed magnitudes
        
        return adjacency_matrix, X, y
    
    def _load_synthetic_data(self, dataset_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load synthetic dataset."""
        # Find the dataset name by looking up which config has this path
        dataset_name = None
        for name, config in self.dataset_configs.items():
            if config.get('loader') == self._load_synthetic_data:
                expected_path = os.path.join(self.data_root, config['type'], config['subdir'])
                if expected_path == dataset_path:
                    dataset_name = name
                    break
        
        if dataset_name is None:
            # Fallback: extract from path
            dataset_name = os.path.basename(dataset_path)
        
        config = self.dataset_configs[dataset_name]
        
        # The path structure: synthetic/single_modal/synthetic_single_modal_1000x1000.npz
        data_file_path = os.path.join(dataset_path, config['data_file'])
        
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Synthetic data file not found: {data_file_path}")
        
        # Load the processed synthetic data
        data = np.load(data_file_path, allow_pickle=True)
        
        # Reconstruct sparse adjacency matrix
        A_data = data['A_data']
        A_indices = data['A_indices']
        A_indptr = data['A_indptr']
        A_shape = data['A_shape']
        adjacency_matrix = csr_matrix((A_data, A_indices, A_indptr), shape=A_shape)
        
        # Load node indices (consistent with other datasets)
        X = data['X']  # shape (num_nodes,) - node indices [0, 1, 2, ...]
        
        # Load synthetic function values (normalized)
        y = data['y']  # shape (num_nodes,) - synthetic function values
        
        return adjacency_matrix, X, y
    
    def _load_from_cache(self, cache_path: str) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Load dataset from cache file."""
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['adjacency_matrix'], data['node_indices'], data['target_data']
    
    def _save_to_cache(self, data: Tuple[csr_matrix, np.ndarray, np.ndarray], cache_path: str, dataset_name: str):
        """Save dataset to cache file."""
        adjacency_matrix, X, y = data
        
        # All datasets now have consistent format: (adjacency, node_indices, target_values)
        # Determine data type
        if dataset_name in ['500hpa', '800hpa', '1000hpa', '500hpa_wide', '800hpa_wide', '1000hpa_wide']:
            data_type = 'wind'
        elif dataset_name in ['single_modal', 'multi_modal', 'bimodal', 'community', 'circular']:
            data_type = 'synthetic'
        else:
            data_type = 'graph'
            
        cache_data = {
            'adjacency_matrix': adjacency_matrix,
            'node_indices': X,
            'target_data': y,  # node degrees for graphs, wind speeds for wind, function values for synthetic
            'num_nodes': len(X),
            'num_edges': adjacency_matrix.nnz // 2,
            'density': adjacency_matrix.nnz / (adjacency_matrix.shape[0] * adjacency_matrix.shape[1]),
            'dataset_name': dataset_name,
            'data_type': data_type
        }
        
        print(f"Cached {dataset_name}:")
        print(f"  Nodes: {cache_data['num_nodes']}")
        print(f"  Edges: {cache_data['num_edges']}")
        print(f"  Density: {cache_data['density']:.6f}")
        
        if data_type == 'wind':
            print(f"  Wind speed range: {y.min():.3f} to {y.max():.3f} (normalized)")
        elif data_type == 'synthetic':
            print(f"  Function value range: {y.min():.3f} to {y.max():.3f} (normalized)")
        else:
            print(f"  Degree range: {y.min()} to {y.max()}")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def list_available_datasets(self) -> list:
        """Return list of available datasets."""
        return list(self.dataset_configs.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset without loading it."""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return {
                'num_nodes': data['num_nodes'],
                'num_edges': data['num_edges'],
                'density': data['density'],
                'cached': True
            }
        else:
            return {'cached': False}
    
    def clear_cache(self, dataset_name: str = None):
        """Clear cache for specific dataset or all datasets."""
        if dataset_name:
            # Clear specific dataset
            cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if dataset_name in self._cache:
                del self._cache[dataset_name]
            print(f"Cleared cache for {dataset_name}")
        else:
            # Clear all cache
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
            self._cache.clear()
            print("Cleared all cache")