# ablation_study.py
import sys, os, gc, warnings, subprocess
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import networkx as nx
from tqdm import tqdm

from efficient_graph_gp.graph_kernels import diffusion_kernel, generate_noisy_samples
from efficient_graph_gp.gpflow_kernels import GraphDiffusionKernel, GraphGeneralFastGRFKernel

warnings.filterwarnings('ignore')

# ----------------------------
# CONFIGURATION
# ----------------------------
CONFIG = {
    "mesh_size": 30,
    "beta_sample": 10,
    "noise_std": 0.5,       # true observation noise std used to *generate* Y_noisy
    "training_fraction": 0.3,
    "seeds": [100 + i for i in range(3)],  # 3 different seeds
    "wpns": [1,10,100,1000],
    "ablation_flags": [False, True],        # ablation vs non-ablation
}
RESULTS_DIR = "experiments_dense/ablation/results"


# ----------------------------
# GPU / memory handling
# ----------------------------
def clear_gpu_memory():
    tf.keras.backend.clear_session()
    gc.collect()


# ----------------------------
# Evaluation metrics (observation space)
# ----------------------------
def nlpd_observation_space(y_true_obs, mu_y, var_y):
    """
    Negative Log Predictive Density in observation space.
    Inputs are 1D numpy arrays of same length.
    Uses model.predict_y() outputs: mean and variance *including learned noise*.
    """
    s2 = np.clip(var_y, 1e-12, None)
    logp = -0.5 * (np.log(2 * np.pi * s2) + (y_true_obs - mu_y) ** 2 / s2)
    return -np.mean(logp)


# ----------------------------
# GP inference (exact GP regression)
# ----------------------------
def gp_inference(X_train, Y_train, graph_kernel, noise_std_true=None):
    """
    Fit a GPflow GPR model with the provided kernel.
    Adds a weak LogNormal prior on likelihood variance centered at noise_std_true^2 (if provided).
    Returns the fitted model, its maximized log marginal likelihood, and (obs-space) predictions on X_train (optional downstream).
    """
    model = gpflow.models.GPR(data=(X_train, Y_train), kernel=graph_kernel, mean_function=None)

    # Optional: initialize likelihood variance near the known noise and place a weak LogNormal prior.
    if noise_std_true is not None:
        noise_var_center = float(noise_std_true ** 2)  # center prior at true variance
        # Prior median = exp(loc); scale is multiplicative spread. scale=0.5 ~ ×[0.6, 1.65]
        model.likelihood.variance.prior = tfp.distributions.LogNormal(
            loc=np.log(noise_var_center), scale=0.5
        )
        # Also initialize the parameter near that value (helps optimizer)
        model.likelihood.variance.assign(noise_var_center)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1000))

    lml = float(model.log_marginal_likelihood().numpy())
    return model, lml


# ----------------------------
# Dataset generation
# ----------------------------
def generate_dataset(mesh_size, beta_sample, noise_std, seed):
    """
    Returns:
      adjacency_matrix (dense np.array),
      X: node indices as [[0],[1],...],
      Y: noise-free latent samples ~ N(0, K_true),
      Y_noisy: observed samples with Gaussian noise added
    """
    num_nodes = mesh_size ** 2
    G = nx.grid_2d_graph(mesh_size, mesh_size)
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    K_true = 10 * diffusion_kernel(adjacency_matrix, beta_sample)

    Y = generate_noisy_samples(K_true, noise_std=0, seed=seed)                   # latent f
    Y_noisy = generate_noisy_samples(K_true, noise_std=noise_std, seed=seed)     # y = f + eps

    X = np.arange(num_nodes, dtype=np.float64).reshape(-1, 1)
    return adjacency_matrix, X, Y, Y_noisy


# ----------------------------
# Single experiment for one seed
# ----------------------------
def run_experiment(adjacency_matrix, X, Y, Y_noisy, seed):
    rng = np.random.default_rng(seed)
    num_nodes = len(Y)
    n_train = max(1, int(num_nodes * CONFIG["training_fraction"]))
    train_idx = rng.choice(num_nodes, n_train, replace=False)
    test_idx = np.setdiff1d(np.arange(num_nodes), train_idx)

    # Tensors for GPflow
    X_train = tf.convert_to_tensor(X[train_idx])
    Y_train = tf.convert_to_tensor(Y_noisy[train_idx].reshape(-1, 1))  # TRAIN ON OBSERVED DATA
    X_full  = tf.convert_to_tensor(X)

    results = []

    # Utility to evaluate a fitted model in observation space
    def eval_model(tag, model, lml, wpn=None):
        # Predict *observation* y (includes learned noise)
        mean_y, var_y = model.predict_y(X_full)
        mu = mean_y.numpy().reshape(-1)
        s2 = var_y.numpy().reshape(-1)

        # TEST TARGETS: use observed (noisy) data for observation-space NLPD
        y_test_obs = Y_noisy[test_idx]
        y_pred_obs = mu[test_idx]
        v_pred_obs = s2[test_idx]

        mse = float(np.mean((y_test_obs - y_pred_obs) ** 2))
        rmse = float(np.sqrt(mse))
        nlpd = float(nlpd_observation_space(y_test_obs, y_pred_obs, v_pred_obs))
        noise_var = float(model.likelihood.variance.numpy())

        print(
            f"        {tag} - LML: {lml:.3f}, noise_var: {noise_var:.6f}, "
            f"MSE(obs): {mse:.6f}, RMSE(obs): {rmse:.6f}, NLPD(obs): {nlpd:.6f}",
            flush=True,
        )

        row = {
            'seed': seed,
            'model': tag,
            'wpn': wpn,
            'lml': lml,
            'mse_obs': mse,
            'rmse_obs': rmse,
            'nlpd_obs': nlpd,
            'learned_noise_var': noise_var,
            'n_train': int(n_train),
            'n_test': int(len(test_idx)),
        }
        results.append(row)

    # ---------------- Diffusion kernel ----------------
    print("  Running Diffusion kernel...", flush=True)
    try:
        graph_kernel = GraphDiffusionKernel(adjacency_matrix)
        model, lml = gp_inference(X_train, Y_train, graph_kernel, noise_std_true=CONFIG["noise_std"])
        eval_model('Diffusion', model, lml, wpn=None)
    except Exception as e:
        print(f"    Diffusion failed: {e}", flush=True)
        results.append({
            'seed': seed, 'model': 'Diffusion', 'wpn': None, 'lml': np.nan,
            'mse_obs': np.nan, 'rmse_obs': np.nan, 'nlpd_obs': np.nan,
            'learned_noise_var': np.nan, 'error': str(e)
        })
    finally:
        clear_gpu_memory()

    # ---------------- GRF kernels over wpns & ablation flag ----------------
    for wpn in tqdm(CONFIG["wpns"], desc=f"    GRF wpns (seed={seed})", leave=False):
        for ablation in CONFIG["ablation_flags"]:
            model_name = 'GRF-ablation' if ablation else 'GRF'
            tag = f"{model_name}(wpn={wpn})"
            print(f"\n      Running {tag}...", flush=True)
            try:
                graph_kernel = GraphGeneralFastGRFKernel(
                    adjacency_matrix,
                    walks_per_node=wpn,
                    p_halt=0.01,
                    max_walk_length=10,
                    use_tqdm=False,
                    ablation=ablation,
                )
                model, lml = gp_inference(X_train, Y_train, graph_kernel, noise_std_true=CONFIG["noise_std"])
                eval_model(tag, model, lml, wpn=wpn)
            except Exception as e:
                print(f"        {tag} failed: {e}", flush=True)
                results.append({
                    'seed': seed, 'model': model_name, 'wpn': wpn, 'lml': np.nan,
                    'mse_obs': np.nan, 'rmse_obs': np.nan, 'nlpd_obs': np.nan,
                    'learned_noise_var': np.nan, 'error': str(e)
                })
            finally:
                clear_gpu_memory()

    return results


# ----------------------------
# Parent orchestration
# ----------------------------
def parent_main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_dfs = []
    for seed in tqdm(CONFIG["seeds"], desc="Seeds"):
        print(f"\nRunning seed {seed} in subprocess...", flush=True)
        subprocess.run([sys.executable, "-u", __file__, str(seed)], check=True)
        df = pd.read_csv(f"{RESULTS_DIR}/results_seed_{seed}.csv")
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(f"{RESULTS_DIR}/ablation_results.csv", index=False)
    print(f"All results saved to {RESULTS_DIR}/ablation_results.csv")


# ----------------------------
# Child execution
# ----------------------------
def child_main(seed: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # IMPORTANT: use this seed for the dataset (not a fixed 42)
    adjacency_matrix, X, Y, Y_noisy = generate_dataset(
        CONFIG["mesh_size"], CONFIG["beta_sample"], CONFIG["noise_std"], seed=seed
    )
    results = run_experiment(adjacency_matrix, X, Y, Y_noisy, seed)
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/results_seed_{seed}.csv", index=False)


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        parent_main()
    else:
        child_main(int(sys.argv[1]))
