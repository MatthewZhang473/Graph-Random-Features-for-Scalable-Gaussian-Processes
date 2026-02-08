import os
import gpytorch
from gpytorch import settings as gsettings
from linear_operator import settings

def setup_gpytorch_settings():
    """Configure GPyTorch and Linear Operator settings"""
    settings.verbose_linalg._default = False
    settings._fast_covar_root_decomposition._default = False
    gsettings.max_cholesky_size._global_value = 0
    gsettings.cg_tolerance._global_value = 1e-2
    gsettings.max_lanczos_quadrature_iterations._global_value = 1
    settings.fast_computations.log_prob._state = True
    gsettings.num_trace_samples._global_value = 64
    gsettings.min_preconditioning_size._global_value = 1e10

def create_directories(data_dir, step_matrices_dir, results_dir):
    """Create necessary directories"""
    for dir_path in [data_dir, step_matrices_dir, results_dir]:
        os.makedirs(dir_path, exist_ok=True)