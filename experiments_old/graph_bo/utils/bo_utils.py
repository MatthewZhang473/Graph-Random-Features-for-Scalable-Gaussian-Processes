import torch
import gpytorch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import os, sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)
from efficient_graph_gp_sparse.models import SparseGraphGP
        

class Algorithm(ABC):
    def __init__(self, n_nodes, device):
        self.n_nodes = n_nodes
        self.device = device
    
    @abstractmethod
    def select_next_points(self, X_observed, Y_observed, batch_size):
        pass
    
    @abstractmethod
    def update(self, X_observed, Y_observed):
        pass

class RandomSearch(Algorithm):
    def select_next_points(self, X_observed, Y_observed, batch_size=1):
        return np.random.choice(self.n_nodes, size=batch_size, replace=False)
    def update(self, X_observed, Y_observed):
        pass
    
class BFS(Algorithm):
    """Breadth-First Search algorithm for graph exploration."""
    def __init__(self, A_sparse, n_nodes, device):
        super().__init__(n_nodes, device)
        self.A_sparse = A_sparse
        self.visited = set()
        self.queue = []
    
    def _get_neighbors(self, node):
        """Get neighbors from sparse adjacency matrix"""
        row = self.A_sparse._getrow(node)
        return row.indices.tolist()
    
    def select_next_points(self, X_observed, Y_observed, batch_size=1):
        observed_indices = X_observed.cpu().numpy().flatten().astype(int)
        observed_values = Y_observed.cpu().numpy().flatten()
        
        # If queue is empty, start from a random visited node
        if not self.queue:
            random_start = np.random.choice(observed_indices)
            neighbors = self._get_neighbors(random_start)
            # Add unvisited neighbors to queue
            for neighbor in neighbors:
                if neighbor not in self.visited and neighbor not in self.queue:
                    self.queue.append(neighbor)
        
        # Remove visited nodes from queue
        self.queue = [node for node in self.queue if node not in self.visited]
        
        # Select points from queue (FIFO for BFS)
        selected = []
        for _ in range(min(batch_size, len(self.queue))):
            if self.queue:
                selected.append(self.queue.pop(0))
        
        # If we need more points and queue is empty, expand from random visited nodes
        while len(selected) < batch_size:
            if not self.queue:
                # Pick a random visited node and add its unvisited neighbors
                random_node = np.random.choice(observed_indices)
                neighbors = self._get_neighbors(random_node)
                for neighbor in neighbors:
                    if neighbor not in self.visited and neighbor not in self.queue and neighbor not in selected:
                        self.queue.append(neighbor)
            
            if self.queue:
                selected.append(self.queue.pop(0))
            else:
                # Fallback to random unvisited node
                unvisited = set(range(self.n_nodes)) - self.visited - set(selected)
                if unvisited:
                    selected.append(np.random.choice(list(unvisited)))
                else:
                    break
        
        return selected

    def update(self, X_observed, Y_observed):
        observed_indices = X_observed.cpu().numpy().flatten().astype(int)
        self.visited.update(observed_indices)
        # Remove newly visited nodes from queue
        self.queue = [node for node in self.queue if node not in self.visited]

class DFS(Algorithm):
    """Depth-First Search algorithm for graph exploration."""
    def __init__(self, A_sparse, n_nodes, device):
        super().__init__(n_nodes, device)
        self.A_sparse = A_sparse
        self.visited = set()
        self.stack = []
    
    def _get_neighbors(self, node):
        """Get neighbors from sparse adjacency matrix"""
        row = self.A_sparse._getrow(node)
        return row.indices.tolist()
    
    def select_next_points(self, X_observed, Y_observed, batch_size=1):
        observed_indices = X_observed.cpu().numpy().flatten().astype(int)
        observed_values = Y_observed.cpu().numpy().flatten()
        
        # If stack is empty, start from a random visited node
        if not self.stack:
            random_start = np.random.choice(observed_indices)
            neighbors = self._get_neighbors(random_start)
            # Add unvisited neighbors to stack
            for neighbor in neighbors:
                if neighbor not in self.visited and neighbor not in self.stack:
                    self.stack.append(neighbor)
        
        # Remove visited nodes from stack
        self.stack = [node for node in self.stack if node not in self.visited]
        
        # Select points from stack (LIFO for DFS)
        selected = []
        for _ in range(min(batch_size, len(self.stack))):
            if self.stack:
                selected.append(self.stack.pop())
        
        # If we need more points and stack is empty, expand from random visited nodes
        while len(selected) < batch_size:
            if not self.stack:
                # Pick a random visited node and add its unvisited neighbors
                random_node = np.random.choice(observed_indices)
                neighbors = self._get_neighbors(random_node)
                for neighbor in neighbors:
                    if neighbor not in self.visited and neighbor not in self.stack and neighbor not in selected:
                        self.stack.append(neighbor)
            
            if self.stack:
                selected.append(self.stack.pop())
            else:
                # Fallback to random unvisited node
                unvisited = set(range(self.n_nodes)) - self.visited - set(selected)
                if unvisited:
                    selected.append(np.random.choice(list(unvisited)))
                else:
                    break
        
        return selected

    def update(self, X_observed, Y_observed):
        observed_indices = X_observed.cpu().numpy().flatten().astype(int)
        self.visited.update(observed_indices)
        # Remove newly visited nodes from stack
        self.stack = [node for node in self.stack if node not in self.visited]

class GreedySearch(Algorithm):
    def __init__(self, A_sparse, n_nodes, device):
        super().__init__(n_nodes, device)
        self.A_sparse = A_sparse
        self.visited = set()
        self.frontier = set()
    
    def _get_neighbors(self, node):
        """Get neighbors from sparse adjacency matrix"""
        row = self.A_sparse._getrow(node)
        return row.indices.tolist()
    
    def select_next_points(self, X_observed, Y_observed, batch_size=1):
        observed_indices = X_observed.cpu().numpy().flatten().astype(int)
        observed_values = Y_observed.cpu().numpy().flatten()
        
        # If no frontier, start from best observed point
        if not self.frontier:
            best_node = observed_indices[np.argmax(observed_values)]
            neighbors = self._get_neighbors(best_node)
            self.frontier.update(n for n in neighbors if n not in self.visited)
        
        # Remove visited nodes from frontier
        self.frontier -= self.visited
        
        # If frontier is empty, expand from current best points
        if len(self.frontier) < batch_size:
            best_indices = np.argsort(observed_values)[-3:]
            for idx in best_indices:
                node = observed_indices[idx]
                neighbors = self._get_neighbors(node)
                self.frontier.update(n for n in neighbors if n not in self.visited)
        
        # Remove visited nodes again
        self.frontier -= self.visited
        
        # Select points from frontier
        if self.frontier:
            selected = list(self.frontier)[:batch_size]
            self.frontier -= set(selected)
            return selected
        else:
            # Fallback to random
            unvisited = set(range(self.n_nodes)) - self.visited
            if unvisited:
                return np.random.choice(list(unvisited), min(batch_size, len(unvisited)), replace=False).tolist()
            else:
                return np.random.choice(self.n_nodes, batch_size, replace=False).tolist()
    
    def update(self, X_observed, Y_observed):
        observed_indices = X_observed.cpu().numpy().flatten().astype(int)
        self.visited.update(observed_indices)
        self.frontier -= self.visited

class SparseGRF(Algorithm):
    def __init__(self, n_nodes, device, step_matrices, max_walk_length, learning_rate, train_epochs, retrain_interval):
        super().__init__(n_nodes, device)
        self.step_matrices = step_matrices
        self.max_walk_length = max_walk_length
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.retrain_interval = retrain_interval
        self.cached_model = None
        self.cached_likelihood = None
        self.last_training_size = 0
        self.visited = set()
    
    def reset_cache(self):
        self.cached_model = None
        self.cached_likelihood = None
        self.last_training_size = 0
    
    def _should_retrain(self, current_size):
        return (self.cached_model is None or 
                self.retrain_interval == 0 or 
                (current_size - self.last_training_size) >= self.retrain_interval)
    
    def _train_model(self, X_observed, Y_observed):
        # Import the sparse GP model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = SparseGraphGP(X_observed, Y_observed, likelihood, self.step_matrices, self.max_walk_length).to(self.device)
        
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        for _ in range(self.train_epochs):
            optimizer.zero_grad()
            output = model(X_observed)
            loss = -mll(output, Y_observed)
            loss.backward()
            optimizer.step()
        
        self.cached_model, self.cached_likelihood = model, likelihood
        self.last_training_size = len(X_observed)
        return model, likelihood

    def select_next_points(self, X_observed, Y_observed, batch_size=1):
        observed_indices = X_observed.cpu().numpy().flatten().astype(int)
        current_size = len(X_observed)
        if self._should_retrain(current_size):
            tqdm.write(f"      Retraining GP with {current_size} points...")
            model, likelihood = self._train_model(X_observed, Y_observed)
        else:
            model, likelihood = self.cached_model, self.cached_likelihood
        
        model.eval()
        likelihood.eval()
        
        unvisited_nodes = list(set(range(self.n_nodes)) - self.visited)
        X_unvisited = torch.tensor(unvisited_nodes, dtype=torch.float32, device=self.device).unsqueeze(1)
        with torch.no_grad():
            thompson_samples = model.predict(X_unvisited, n_samples=1)  # shape (1, n_unvisited)
            topk = torch.topk(thompson_samples[0, :], batch_size)
            selected_indices = [unvisited_nodes[idx] for idx in topk.indices.tolist()]

        return selected_indices

    def update(self, X_observed, Y_observed):
        observed_indices = X_observed.cpu().numpy().flatten().astype(int)
        self.visited.update(observed_indices)
        self.cached_model.x_train = X_observed
        self.cached_model.y_train = Y_observed

class BayesianOptimizer:
    def __init__(self, algorithm, objective_values, initial_points=10, batch_size=1):
        self.algorithm = algorithm
        self.objective_values = objective_values
        self.n_nodes = len(objective_values)
        self.initial_points = initial_points
        self.batch_size = batch_size
        self.gt_best_value = float(objective_values[np.argmax(objective_values)])
    
    def run_optimization(self, n_iterations, seed=None, algorithm_name="BO"):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        results = []
        
        # Initial random selection
        observed_indices = np.random.choice(self.n_nodes, self.initial_points, replace=False)
        
        X_observed = torch.tensor(observed_indices.reshape(-1, 1), dtype=torch.float32, device=self.algorithm.device)
        Y_observed = torch.tensor(self.objective_values[observed_indices].flatten(), dtype=torch.float32, device=self.algorithm.device)
        
        best_value = float(Y_observed.max())
        best_idx = observed_indices[torch.argmax(Y_observed).item()]
        
        with tqdm(range(n_iterations), desc=f"    {algorithm_name}", leave=False) as pbar:
            for iteration in pbar:
                # Select next points
                next_indices = self.algorithm.select_next_points(X_observed, Y_observed, self.batch_size)
                
                # Evaluate and update
                batch_values = []
                for next_idx in next_indices:
                    next_value = float(self.objective_values[next_idx])
                    if next_value > best_value:
                        best_value = next_value
                        best_idx = next_idx
                    batch_values.append(next_value)
                    observed_indices = np.append(observed_indices, next_idx)
                
                # Update tensors
                X_observed = torch.tensor(observed_indices.reshape(-1, 1), dtype=torch.float32, device=self.algorithm.device)
                Y_observed = torch.tensor(self.objective_values[observed_indices].flatten(), dtype=torch.float32, device=self.algorithm.device)
                
                # Update algorithm
                self.algorithm.update(X_observed, Y_observed)
                
                # Record results
                results.append({
                    'iteration': iteration + 1,
                    'best_value': best_value,
                    'best_point': best_idx,
                    'regret': self.gt_best_value - best_value,
                    'dataset_size': len(observed_indices),
                    'batch_mean': np.mean(batch_values),
                    'batch_max': np.max(batch_values),
                })
                
                pbar.set_postfix({
                    'best': f'{best_value:.4f}',
                    'regret': f'{(self.gt_best_value - best_value):.4f}',
                    'points': len(observed_indices)
                })
        
        return results
