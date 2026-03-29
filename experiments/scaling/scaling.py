import os
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import gpytorch
from linear_operator import settings
from gpytorch import settings as gsettings

project_root = Path(__file__).resolve().parents[2]
venv_site_packages = (
    project_root
    / "venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
for path in (project_root, venv_site_packages):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from grf_gp.kernels.general import GeneralGRFKernel
from grf_gp.model import GRFGP
from grf_gp.sampler import GRFSampler
from grf_gp.utils.config import set_gp_defaults

from experiments.utils import train_model


CONFIG = {
    "graph_sizes": [2**i for i in range(6, 11)],
    "seeds": [0, 1, 2],
    "train_fraction": 0.6,
    "noise_std": 0.1,
    "walks_per_node": 1_000,
    "p_halt": 0.1,
    "max_walk_length": 5,
    "train_lr": 0.05,
    "train_iters": 100,
    "predict_samples": 64,
    "n_processes": 4,
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
set_gp_defaults(linear_operator_settings=settings, gpytorch_settings=gsettings)


def sparse_linear_operator_storage_mb(op):
    tensor = op.sparse_csr_tensor
    value_bytes = tensor.values().element_size() * tensor.values().numel()
    crow_bytes = tensor.crow_indices().element_size() * tensor.crow_indices().numel()
    col_bytes = tensor.col_indices().element_size() * tensor.col_indices().numel()
    return (value_bytes + crow_bytes + col_bytes) / 1024 / 1024


def make_graph_data(n_nodes, seed):
    rng = np.random.default_rng(seed)
    graph = nx.cycle_graph(n_nodes)
    adjacency = nx.to_numpy_array(graph, dtype=np.float64)
    x_all = np.arange(n_nodes)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    y_true = 2.0 * np.sin(2 * angles) + 0.5 * np.cos(4 * angles) + 0.3 * np.sin(angles)
    y_obs = y_true + rng.normal(0.0, CONFIG["noise_std"], size=n_nodes)

    n_train = int(CONFIG["train_fraction"] * n_nodes)
    train_idx = rng.choice(n_nodes, size=n_train, replace=False)
    test_idx = np.setdiff1d(x_all, train_idx)
    return adjacency, x_all, y_obs, train_idx, test_idx


def run_single_experiment(n_nodes, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    adjacency, x_all, y_obs, train_idx, test_idx = make_graph_data(n_nodes, seed)
    adjacency_torch = torch.tensor(adjacency, dtype=dtype, device=device)

    x_train = torch.tensor(train_idx, dtype=torch.long, device=device)
    x_test = torch.tensor(test_idx, dtype=torch.long, device=device)
    y_train_raw = torch.tensor(y_obs[train_idx], dtype=dtype, device=device)

    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std()
    y_train = ((y_train_raw - y_mean) / y_std).flatten()

    row = {
        "n_nodes": n_nodes,
        "n_edges": graph_edge_count(n_nodes),
        "seed": seed,
        "device": str(device),
    }

    t0 = time.perf_counter()
    sampler = GRFSampler(
        adjacency_torch,
        walks_per_node=CONFIG["walks_per_node"],
        p_halt=CONFIG["p_halt"],
        max_walk_length=CONFIG["max_walk_length"],
        seed=seed,
        use_tqdm=False,
        n_processes=CONFIG["n_processes"],
    )
    rw_mats = sampler()
    row["sampling_time_sec"] = time.perf_counter() - t0
    row["sampling_storage_mb"] = sum(
        sparse_linear_operator_storage_mb(op) for op in rw_mats
    )
    row["sampling_total_nnz"] = sum(
        op.sparse_csr_tensor.values().numel() for op in rw_mats
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
        device=device, dtype=dtype
    )
    kernel = GeneralGRFKernel(rw_mats, CONFIG["max_walk_length"]).to(
        device=device, dtype=dtype
    )
    model = GRFGP(x_train, y_train, likelihood, kernel).to(device=device, dtype=dtype)
    t0 = time.perf_counter()
    train_model(
        model,
        likelihood,
        x_train,
        y_train,
        lr=CONFIG["train_lr"],
        max_iter=CONFIG["train_iters"],
        progress_desc=f"train n={n_nodes} seed={seed}",
    )
    row["training_time_sec"] = time.perf_counter() - t0

    model.eval()
    likelihood.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model.predict(x_test, batch_size=CONFIG["predict_samples"])
        _ = pred.mean.detach().cpu()
    row["inference_time_sec"] = time.perf_counter() - t0

    row["noise"] = likelihood.noise.item()
    row["modulation_norm"] = kernel.modulation_function.norm().item()
    return row


def graph_edge_count(n_nodes):
    return n_nodes


def main():
    rows = []
    for n_nodes in CONFIG["graph_sizes"]:
        for seed in CONFIG["seeds"]:
            print(f"Running n_nodes={n_nodes}, seed={seed}")
            rows.append(run_single_experiment(n_nodes=n_nodes, seed=seed))

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"scaling_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    print(df.head())


if __name__ == "__main__":
    main()
