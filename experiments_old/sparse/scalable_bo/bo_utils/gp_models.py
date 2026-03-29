import torch
import gpytorch
from gpytorch import settings as gsettings
from linear_operator.utils import linear_cg
from linear_operator.operators import IdentityLinearOperator
import sys
import os

# Add the correct path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from efficient_graph_gp_sparse.gptorch_kernels_sparse.sparse_grf_kernel import SparseGRFKernel

class SparseGraphGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, step_matrices, max_walk_length):
        super().__init__(x_train, y_train, likelihood)
        self.x_train, self.y_train = x_train, y_train
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = SparseGRFKernel(max_walk_length=max_walk_length, step_matrices_torch=step_matrices)
        self.num_nodes = step_matrices[0].shape[0]
        
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))
    
    def predict(self, x_test, n_samples=64):
        train_indices = self.x_train.int().flatten()
        test_indices = x_test.int().flatten()
        
        phi = self.covar_module._get_feature_matrix()
        phi_train = phi[train_indices, :]
        phi_test = phi[test_indices, :]
        
        K_train_train = phi_train @ phi_train.T
        K_test_train = phi_test @ phi_train.T
        
        noise_variance = self.likelihood.noise.item()
        noise_std = torch.sqrt(torch.tensor(noise_variance, device=x_test.device))
        A = K_train_train + noise_variance * IdentityLinearOperator(len(train_indices), device=x_test.device)
        
        eps1_batch = torch.randn(n_samples, self.num_nodes, device=x_test.device)
        eps2_batch = noise_std * torch.randn(n_samples, len(train_indices), device=x_test.device)
        
        f_test_prior = eps1_batch @ phi_test.T
        f_train_prior = eps1_batch @ phi_train.T
        
        b_batch = self.y_train.unsqueeze(0) - (f_train_prior + eps2_batch)
        v_batch = linear_cg(A._matmul, b_batch.T, tolerance=gsettings.cg_tolerance.value())
        
        return f_test_prior + (K_test_train @ v_batch).T
