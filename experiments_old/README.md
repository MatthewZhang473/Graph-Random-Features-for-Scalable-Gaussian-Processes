# Experiments (paper-only)

This folder contains only the experiments discussed in *Graph Random Features for Scalable Gaussian Processes*.

## Dense (GPflow)
- `traffic_dataset/`: San Jose freeway speed regression. Includes data loaders, notebooks, and scripts for diffusion/GRF/PoFM baselines.
- `cora/`: Cora citation classification with diffusion/Matérn/GRF kernels.
- `ablation/`: Random-walk kernel ablation (ad-hoc vs principled GRF) used in Appendix analyses.

## Sparse (GPyTorch)
- `scaling_exp/`: Scaling benchmarks demonstrating O(N^{3/2}) time via CG on GRF kernels.
- `scalable_bo/`: Bayesian optimisation on synthetic grids, social networks, and wind fields. Contains configs, scripts, processed results, and plotting notebooks.
- `social_networks/`: SNAP graph assets used by the BO experiments.
- `scripts/`: Utilities for sparse graph GP experiments.

## BO code
`graph_bo/` (top level) holds the BO runners and configs referenced in the paper. Use the configs under `graph_bo/configs/` and scripts under `graph_bo/scripts/`.

## Data
Large datasets (PEMS traffic, SNAP social graphs, ERA5 wind) are not checked in. Use the loaders/notebooks within each subfolder to download/preprocess the data before running experiments.

## How to run
- Install deps: `pip install -r ../requirements.txt` from repo root.
- For each subfolder, consult its README/notebooks for exact commands. Most scripts/notebooks assume data has been downloaded via the provided loaders.

