import os
import yaml
from typing import Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    """Configuration class for Graph BO experiments"""
    
    # Experiment settings
    datasets: List[str] = field(default_factory=lambda: ['facebook'])
    algorithms: List[str] = field(default_factory=lambda: ['random_search', 'bfs', 'sparse_grf'])
    
    # BO parameters
    iterations: int = 10
    runs: int = 3
    initial_points: int = 100
    batch_size: int = 50
    seeds: List[int] = field(default_factory=lambda: [100, 110, 120])
    
    # GRF parameters
    walks_per_node: int = 10000
    p_halt: float = 0.1
    max_walk_length: int = 3
    
    # Training parameters
    learning_rate: float = 0.01
    train_epochs: int = 30
    gp_retrain_interval: int = 300
    
    # Directory paths
    step_matrices_dir: str = ""
    results_dir: str = ""
    
    # Random seeds
    numpy_seed: int = 42
    torch_seed: int = 42
    random_walk_seed: int = 42
    
    def __post_init__(self):
        """Setup derived attributes after initialization"""
        # Auto-generate seeds if not provided or if runs changed
        if len(self.seeds) != self.runs:
            self.seeds = [100 + i * 10 for i in range(self.runs)]
        
        # Setup directory paths if not absolute
        if not os.path.isabs(self.step_matrices_dir):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(base_dir, '..', '..')
            self.step_matrices_dir = os.path.join(project_root, self.step_matrices_dir)
        
        if not os.path.isabs(self.results_dir):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(base_dir, '..', '..')
            self.results_dir = os.path.join(project_root, self.results_dir)
        
        # Create directories
        os.makedirs(self.step_matrices_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

def load_config_from_yaml(config_path: str) -> ExperimentConfig:
    """
    Load configuration from YAML file and return ExperimentConfig object
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig object with loaded parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    
    # Flatten nested structure for easier access
    config_dict = {}
    
    # Top-level settings
    config_dict['datasets'] = yaml_config.get('datasets', ['facebook'])
    config_dict['algorithms'] = yaml_config.get('algorithms', ['random_search', 'bfs', 'sparse_grf'])
    
    # BO parameters
    bo_params = yaml_config.get('bo_parameters', {})
    config_dict['iterations'] = bo_params.get('iterations', 10)
    config_dict['runs'] = bo_params.get('runs', 3)
    config_dict['initial_points'] = bo_params.get('initial_points', 100)
    config_dict['batch_size'] = bo_params.get('batch_size', 50)
    config_dict['seeds'] = bo_params.get('seeds', [100, 110, 120])
    
    # GRF parameters
    grf_params = yaml_config.get('grf_parameters', {})
    config_dict['walks_per_node'] = grf_params.get('walks_per_node', 10000)
    config_dict['p_halt'] = grf_params.get('p_halt', 0.1)
    config_dict['max_walk_length'] = grf_params.get('max_walk_length', 3)
    
    # Training parameters
    train_params = yaml_config.get('training_parameters', {})
    config_dict['learning_rate'] = train_params.get('learning_rate', 0.01)
    config_dict['train_epochs'] = train_params.get('train_epochs', 30)
    config_dict['gp_retrain_interval'] = train_params.get('gp_retrain_interval', 300)
    
    # Directory paths
    directories = yaml_config.get('directories', {})
    config_dict['step_matrices_dir'] = directories.get('step_matrices', 'graph_bo/data/step_matrices')
    config_dict['results_dir'] = directories.get('results', 'graph_bo/results')
    
    # Random seeds
    random_seeds = yaml_config.get('random_seeds', {})
    config_dict['numpy_seed'] = random_seeds.get('numpy_seed', 42)
    config_dict['torch_seed'] = random_seeds.get('torch_seed', 42)
    config_dict['random_walk_seed'] = random_seeds.get('random_walk_seed', 42)
    
    return ExperimentConfig(**config_dict)

def get_default_config_path() -> str:
    """Get the path to the default configuration file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, '..', 'configs', 'default_config.yaml')