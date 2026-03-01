# Graph Random Features for Scalable Gaussian Processes

This repository contains the official implementation and experiment suite for the paper **"Graph Random Features for Scalable Gaussian Processes"** ([arXiv:2509.03691](https://arxiv.org/abs/2509.03691)). 



## Abstract

We study the application of graph random features (GRFs) - a recently introduced stochastic estimator of graph node kernels - to scalable Gaussian processes on discrete input spaces. We prove that (under mild assumptions) Bayesian inference with GRFs enjoys O(N3/2) time complexity with respect to the number of nodes N, compared to O(N3) for exact kernels. Substantial wall-clock speedups and memory savings unlock Bayesian optimisation on graphs with over 106 nodes on a single computer chip, whilst preserving competitive performance.


## Core Implementation

The machinery for the GRF-based Gaussian Processes is maintained in our core library:

> **[Fast-Graph-GP](https://github.com/MatthewZhang473/Fast-Graph-GP)**: A package for performing fast Gaussian Process (GP) inference on graphs. Internally, it uses Graph Random Features (GRFs) to compute a unbiased & sparse estimate of a family of well-known graph node kernels. It further uses path-wise conditioning to leverage the sparsity of the kernel approximation, enabling you to perform GP model train / inference in $\mathcal{O}(N^{3/2})$ time and $\mathcal{O}(N)$ space complexity.

## Reproducing Experiments

We are currently refactoring the experiment Jupyter notebooks to improve accessibility. You can follow the progress or check out the updated code in the active Pull Request:

* **[PR #2: Refactoring experiment notebooks](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes/pull/2)**