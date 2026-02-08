import torch
import os, sys
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)
from efficient_graph_gp_sparse.preprocessor import GraphPreprocessor
from efficient_graph_gp_sparse.utils_sparse import SparseLinearOperator

def convert_step_matrices_to_device(step_matrices_scipy, device):
    result = []
    for mat in step_matrices_scipy:
        tensor = GraphPreprocessor.from_scipy_csr(mat).to(device)
        result.append(SparseLinearOperator(tensor))
    return result

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cleanup_gpu_memory():
    """Clean up GPU memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
