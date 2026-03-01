# Graph Bayesian Optimisation (paper experiments)

Minimal BO runners/configs used in the paper:
- Synthetic benchmarks
- SNAP social networks (YouTube, Twitch, Facebook, Enron)
- Wind magnitude fields (ERA5 variants)

## How to run
From repo root:
```bash
python graph_bo/scripts/run_graph_bo.py --config graph_bo/configs/social_networks_small.yaml
python graph_bo/scripts/run_graph_bo.py --config graph_bo/configs/synthetic_0917.yaml
python graph_bo/scripts/run_graph_bo.py --config graph_bo/configs/wind_magnitude.yaml
```
Configs are YAML and self-describing. `default_config.yaml` points to a small social set; `wind_magnitude_downsampled.yaml` is a lighter wind config.

## What’s inside
- `configs/`: curated configs for the paper (synthetic, social, wind). Extraneous/dated configs removed.
- `scripts/run_graph_bo.py`: main entrypoint for BO across datasets/algorithms.
- `utils/`: data loading, config parsing, algorithm implementations, device helpers.
- `data/`: raw/processed data folders (download via the loaders in `data/raw_data/**` notebooks or scripts).
- `notebooks/`: minimal plotting notebooks to reproduce figures (`plot_social.ipynb`, `plot_synthetic_0917.ipynb`, `plot_wind.ipynb`, `BO_combined_visuals.ipynb`; loader sanity check: `test_data_loader.ipynb`).

## Notes
- BO algorithms: Random Search, BFS/DFS baselines, Greedy Search, and Sparse GRF-GP Thompson sampling (as in the paper).
- Acquisition: Thompson sampling.
- Outputs: results CSVs saved under `graph_bo/results` (set in configs); step matrices cached under `graph_bo/data/step_matrices`.

