# Graph Random Features for Scalable Gaussian Processes

This repository contains the official implementation and experiment suite for the paper **"Graph Random Features for Scalable Gaussian Processes"** ([arXiv:2509.03691](https://arxiv.org/abs/2509.03691)). 



## Abstract

We study the application of graph random features (GRFs) - a recently introduced stochastic estimator of graph node kernels - to scalable Gaussian processes on discrete input spaces. We prove that (under mild assumptions) Bayesian inference with GRFs enjoys O(N3/2) time complexity with respect to the number of nodes N, compared to O(N3) for exact kernels. Substantial wall-clock speedups and memory savings unlock Bayesian optimisation on graphs with over 106 nodes on a single computer chip, whilst preserving competitive performance.


## Core Implementation

The machinery for the GRF-based Gaussian Processes is maintained in our core library:

> **[Fast-Graph-GP](https://github.com/MatthewZhang473/Fast-Graph-GP)**: A package for performing fast Gaussian Process (GP) inference on graphs. Internally, it uses Graph Random Features (GRFs) to compute a unbiased & sparse estimate of a family of well-known graph node kernels. It further uses path-wise conditioning to leverage the sparsity of the kernel approximation, enabling you to perform GP model train / inference in $\mathcal{O}(N^{3/2})$ time and $\mathcal{O}(N)$ space complexity.

## Refactor Progress

We are actively refactoring the code for every experiments to improve accessibility and reproducibility. The table below summarises what is already available and what is still in progress.

| Area | Status | Resources |
| --- | --- | --- |
| GRF-GP inference engine | Complete | Core implementation is maintained in [Fast-Graph-GP](https://github.com/MatthewZhang473/Fast-Graph-GP) |
| Regression: traffic speed prediction | Complete | [demo](experiments/regression/traffic_prediction/demo.ipynb), [experiment](experiments/regression/traffic_prediction/sweep.ipynb), [visualisation](experiments/regression/traffic_prediction/plot.ipynb) |
| Regression: wind interpolation | Complete | [demo](experiments/regression/wind_interpolation/demo.ipynb), [experiment](experiments/regression/wind_interpolation/sweep.ipynb), [visualisation](experiments/regression/wind_interpolation/plot.ipynb) |
| Bayesian optimisation: social networks | Complete | [demo](experiments/bayesopt/social_networks/demo.ipynb), [experiment](experiments/bayesopt/social_networks/sweep.ipynb), [visualization](experiments/bayesopt/social_networks/plot.ipynb)|
| Scaling experiments | Completed | [experiment](experiments/scaling/scaling.py), [visualization](experiments/scaling/plot.ipynb) |
| Ablation studies | Completed | [demo_1](experiments/ablation/2d_mesh.ipynb), [demo_2](experiments/ablation/synthetic_graph.ipynb) |

## Citation

If this repository or the accompanying paper is useful in your work, please cite:

```bibtex
@article{zhang2025graph,
  title={Graph Random Features for Scalable Gaussian Processes},
  author={Zhang, Matthew and Lin, Jihao Andreas and Choromanski, Krzysztof and Weller, Adrian and Turner, Richard E. and Reid, Isaac},
  journal={arXiv preprint arXiv:2509.03691},
  year={2025}
}
```
