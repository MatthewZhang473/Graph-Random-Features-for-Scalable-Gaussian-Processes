from collections import deque

import gpytorch
import numpy as np
import torch

from grf_gp.kernels.general import GeneralGRFKernel
from grf_gp.model import GRFGP

from experiments.utils import train_model


def build_neighbor_lists(adjacency, num_nodes):
    return [
        adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]]
        for i in range(num_nodes)
    ]


def unobserved_nodes(x, observed_set):
    observed = np.fromiter(observed_set, dtype=np.int64)
    return np.setdiff1d(x, observed, assume_unique=False)


def random_search(observed, observed_set, state, rng, context, batch_size):
    del observed, state
    pool = unobserved_nodes(context["X"], observed_set)
    batch_size = min(batch_size, len(pool))
    return rng.choice(pool, size=batch_size, replace=False).tolist()


def bfs(observed, observed_set, state, rng, context, batch_size):
    del observed
    batch = []
    frontier = state["bfs_frontier"]
    while frontier and len(batch) < batch_size:
        node = frontier.popleft()
        if node not in observed_set:
            batch.append(node)
    if len(batch) < batch_size:
        extra = random_search(
            observed=None,
            observed_set=observed_set.union(batch),
            state=state,
            rng=rng,
            context=context,
            batch_size=batch_size - len(batch),
        )
        batch.extend(extra)
    return batch


def dfs(observed, observed_set, state, rng, context, batch_size):
    del observed
    batch = []
    frontier = state["dfs_frontier"]
    while frontier and len(batch) < batch_size:
        node = frontier.pop()
        if node not in observed_set:
            batch.append(node)
    if len(batch) < batch_size:
        extra = random_search(
            observed=None,
            observed_set=observed_set.union(batch),
            state=state,
            rng=rng,
            context=context,
            batch_size=batch_size - len(batch),
        )
        batch.extend(extra)
    return batch


def extend_frontier(frontier, nodes, neighbors):
    for node in nodes:
        frontier.extend(int(neighbor) for neighbor in neighbors[node])


class GRFThompson:
    def __init__(
        self,
        context,
        max_walk_length,
        batch_size,
        retrain_interval,
        train_lr,
        train_iters,
        device,
    ):
        self.context = context
        self.max_walk_length = max_walk_length
        self.batch_size = batch_size
        self.retrain_interval = retrain_interval
        self.train_lr = train_lr
        self.train_iters = train_iters
        self.device = device
        self.model = None
        self.last_fit_size = 0

    def fit(self, observed):
        x_train = torch.tensor(observed, dtype=torch.long, device=self.device)
        y_train = torch.tensor(
            self.context["Y_norm"][observed], dtype=torch.float64, device=self.device
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=self.device, dtype=torch.float64
        )
        kernel = GeneralGRFKernel(
            self.context["rw_mats"], self.max_walk_length
        ).to(device=self.device, dtype=torch.float64)
        self.model = GRFGP(x_train, y_train, likelihood, kernel).to(
            device=self.device, dtype=torch.float64
        )
        train_model(
            self.model,
            likelihood,
            x_train,
            y_train,
            lr=self.train_lr,
            max_iter=self.train_iters,
            progress_desc="GRF fit",
        )
        self.model.eval()
        self.last_fit_size = len(observed)

    def __call__(self, observed, observed_set, state, rng, context, batch_size):
        del state, context
        pool = unobserved_nodes(self.context["X"], observed_set)
        batch_size = min(batch_size, len(pool))
        if len(observed) == 0:
            return rng.choice(pool, size=batch_size, replace=False).tolist()

        needs_refit = self.model is None or (
            len(observed) - self.last_fit_size
        ) >= self.retrain_interval
        if needs_refit:
            self.fit(observed)

        x_pool = torch.tensor(pool, dtype=torch.long, device=self.device)
        with torch.no_grad():
            sample = self.model.predict_sample(x_pool, n_samples=1).squeeze(0)
        order = torch.argsort(sample, descending=True)
        return x_pool[order[:batch_size]].cpu().tolist()


def make_bo_state():
    return {
        "bfs_frontier": deque(),
        "dfs_frontier": [],
    }


def run_bo(name, select_batch, context, n_bo_steps, batch_size, seed, verbose=True):
    rng = np.random.default_rng(seed)
    observed = []
    observed_set = set()
    state = make_bo_state()
    records = []

    for step in range(1, n_bo_steps + 1):
        batch = select_batch(observed, observed_set, state, rng, context, batch_size)
        observed.extend(batch)
        observed_set.update(batch)
        extend_frontier(state["bfs_frontier"], batch, context["neighbors"])
        extend_frontier(state["dfs_frontier"], batch, context["neighbors"])

        best_value = float(context["Y"][observed].max())
        regret = float(context["ground_truth_best"] - best_value)
        if verbose:
            print(
                f"{name:15s} step {step:02d} | batch={len(batch):3d} | "
                f"visited={len(observed):4d} | best={best_value:8.1f} | "
                f"regret={regret:8.1f}"
            )
        records.append(
            {
                "step": step,
                "visited": len(observed),
                "best_value": best_value,
                "regret": regret,
            }
        )

    return {
        "observed_nodes": observed,
        "records": records,
    }
