import gpytorch
from gpytorch import settings as gsettings

def setup_gpytorch_settings():
    """Configure GPyTorch settings for sparse GP training"""
    gsettings.max_cholesky_size._global_value = 0
    gsettings.cg_tolerance._global_value = 1e-2
    gsettings.max_lanczos_quadrature_iterations._global_value = 1
    gsettings.num_trace_samples._global_value = 64
    gsettings.min_preconditioning_size._global_value = 1e10
