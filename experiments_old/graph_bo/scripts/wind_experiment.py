# wind_experiment.py
import sys, os, warnings, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gpytorch
from gpytorch import settings as gsettings
from linear_operator import settings
from linear_operator.utils import linear_cg
from linear_operator.operators import IdentityLinearOperator
import networkx as nx
from scipy import sparse
from netCDF4 import Dataset
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, wgs84, utc

warnings.filterwarnings("ignore")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from efficient_graph_gp_sparse.gptorch_kernels_sparse import SparseGRFKernel, SparseDiffusionKernel
from efficient_graph_gp_sparse.preprocessor import GraphPreprocessor

# ----------------------------
# CONFIGURATION
# ----------------------------
CONFIG = {
    "seeds": [0, 1, 2],
    "walks_per_nodes": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "models": ["SparseGRF", "SparseDiffusion"],  # Test both kernels
    "num_iterations": 1000,
    "learning_rate": 0.01,
    "print_interval": 100,
    "pathwise_samples": 200,
    "downsample_factor": 10,
    "use_downsampling": True,
    "p_halt": 0.1,
    "max_walk_length": 5,
    "nc_file": "graph_bo/data/raw_data/wind_interpolation/8176c14c59fd8dc32a74a89b926cb7fd.nc"
}

RESULTS_DIR = "graph_bo/results"

# GPyTorch / linear_operator settings
settings.verbose_linalg._default = False
settings._fast_covar_root_decomposition._default = False
gsettings.max_cholesky_size._global_value = 0
gsettings.cg_tolerance._global_value = 1e-2
gsettings.max_lanczos_quadrature_iterations._global_value = 1
settings.fast_computations.log_prob._state = True
gsettings.num_trace_samples._global_value = 64
gsettings.min_preconditioning_size._global_value = 1e10

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# Utility functions
# ----------------------------
def deg2rad(x):
    return np.deg2rad(x)

def sph2cart(lat_deg, lon_deg, r=1.0):
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack([x, y, z], axis=-1)

def great_circle_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg, R=1.0):
    lat1, lon1 = deg2rad(lat1_deg), deg2rad(lon1_deg)
    lat2, lon2 = deg2rad(lat2_deg), deg2rad(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def grid_index(i, j, n_lat, n_lon):
    return i * n_lon + j

def inverse_grid_index(node_id, n_lat, n_lon):
    i = node_id // n_lon
    j = node_id % n_lon
    return i, j

def build_sphere_grid_graph(lat, lon, connectivity=4, weight="geodesic", radius=1.0):
    n_lat = len(lat)
    n_lon = len(lon)

    Lon_grid, Lat_grid = np.meshgrid(lon, lat)
    xyz = sph2cart(Lat_grid.ravel(), Lon_grid.ravel(), r=1.0)

    rows, cols, data = [], [], []
    nbrs_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    nbrs_8 = nbrs_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    nbrs = nbrs_8 if connectivity == 8 else nbrs_4

    for i in range(n_lat):
        for j in range(n_lon):
            nid = grid_index(i, j, n_lat, n_lon)
            for di, dj in nbrs:
                ii = i + di
                jj = (j + dj) % n_lon
                if 0 <= ii < n_lat:
                    nid2 = grid_index(ii, jj, n_lat, n_lon)
                    w = great_circle_distance(lat[i], lon[j], lat[ii], lon[jj], R=radius) if weight == "geodesic" else 1.0
                    rows.append(nid); cols.append(nid2); data.append(w)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_lat*n_lon, n_lat*n_lon))
    A = ((A + A.T) * 0.5).tocsr()

    G = nx.from_scipy_sparse_array(A)
    node_attrs = {}
    for nid in range(n_lat * n_lon):
        i, j = inverse_grid_index(nid, n_lat, n_lon)
        node_attrs[nid] = {"lat": float(lat[i]), "lon": float(lon[j]), "xyz": tuple(xyz[nid])}
    nx.set_node_attributes(G, node_attrs)

    return G, A

def downsample_grid_data(lat, lon, u_data, v_data, factor=10):
    lat_down = lat[::factor]
    lon_down = lon[::factor]
    u_down = u_data[::factor, ::factor]
    v_down = v_data[::factor, ::factor]
    return lat_down, lon_down, u_down, v_down

def nearest_node_indices_for_track(track_lat, track_lon, lat, lon):
    lat = np.asarray(lat); lon = np.asarray(lon)
    track_lat = np.asarray(track_lat)
    track_lon = np.asarray(track_lon) % 360.0

    i_idx = np.abs(track_lat[:, None] - lat[None, :]).argmin(axis=1)
    j_idx = np.abs(track_lon[:, None] - lon[None, :]).argmin(axis=1)

    node_ids = i_idx * len(lon) + j_idx
    idx_ij = np.stack([i_idx, j_idx], axis=1)
    return idx_ij, node_ids

def generate_aeolus_track():
    line1 = "1 43600U 18066A   21153.73585495  .00031128  00000-0  12124-3 0  9990"
    line2 = "2 43600  96.7150 160.8035 0006915  90.4181 269.7884 15.87015039160910"

    ts = load.timescale()
    aeolus = EarthSatellite(line1, line2, "AEOLUS", ts)

    start = datetime(2019, 1, 1, 9, tzinfo=utc)
    stop = start + timedelta(hours=24)
    step = timedelta(minutes=1)

    times, t = [], start
    while t <= stop:
        times.append(t)
        t += step

    geocentric = aeolus.at(ts.from_datetimes(times))
    lat, lon = wgs84.latlon_of(geocentric)

    lat = lat.degrees
    lon = (lon.degrees % 360)
    return pd.DataFrame({"time": times, "lat": lat, "lon": lon})

# ----------------------------
# Data preparation
# ----------------------------
def load_and_prepare_data(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load ERA5 wind data (safe close)
    with Dataset(CONFIG["nc_file"], mode="r") as dataset:
        lat = dataset.variables["latitude"][:]
        lon = dataset.variables["longitude"][:]
        u = dataset.variables["u"][:]
        v = dataset.variables["v"][:]

    u_500 = u[0, 0, :, :]
    v_500 = v[0, 0, :, :]

    if CONFIG["use_downsampling"]:
        lat_proc, lon_proc, u_proc, v_proc = downsample_grid_data(
            lat, lon, u_500, v_500, factor=CONFIG["downsample_factor"]
        )
    else:
        lat_proc, lon_proc, u_proc, v_proc = lat, lon, u_500, v_500

    # Orbit track and snapping
    raw_track = generate_aeolus_track()

    def snap_to_grid(lat_val, lon_val):
        i = np.abs(lat_proc - lat_val).argmin()
        j = np.abs(lon_proc - lon_val).argmin()
        return lat_proc[i], lon_proc[j]

    snapped = [snap_to_grid(phi, lam) for phi, lam in zip(raw_track["lat"], raw_track["lon"])]
    snap_lat, snap_lon = zip(*snapped)
    snapped_track = pd.DataFrame({"time": raw_track["time"], "lat": snap_lat, "lon": snap_lon})

    # Graph
    G, A = build_sphere_grid_graph(lat_proc, lon_proc, connectivity=4, weight="geodesic", radius=1.0)

    # Training node ids from orbit
    _, node_ids = nearest_node_indices_for_track(
        track_lat=snapped_track["lat"].values,
        track_lon=snapped_track["lon"].values,
        lat=lat_proc, lon=lon_proc
    )
    unique_train_nodes = np.unique(node_ids)

    # Wind speed targets
    n_lat, n_lon = len(lat_proc), len(lon_proc)
    X = np.arange(n_lat * n_lon)
    y = np.zeros(n_lat * n_lon)
    for i in range(n_lat):
        for j in range(n_lon):
            node_id = i * n_lon + j
            y[node_id] = np.sqrt(u_proc[i, j]**2 + v_proc[i, j]**2)

    # Normalize with training-only stats (no leakage)
    y_mean = float(np.mean(y[unique_train_nodes]))
    y_std = float(np.std(y[unique_train_nodes]))
    y = (y - y_mean) / y_std

    X_train = unique_train_nodes
    y_train = y[X_train]

    return {
        'A': A,
        'X': torch.tensor(X, dtype=torch.long, device=device).unsqueeze(1),
        'y': torch.tensor(y, dtype=torch.float32, device=device),
        'X_train': torch.tensor(X_train, dtype=torch.long, device=device).unsqueeze(1),
        'y_train': torch.tensor(y_train, dtype=torch.float32, device=device),
        'y_mean': y_mean,
        'y_std': y_std
    }

# ----------------------------
# GP Model
# ----------------------------
class GraphGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, step_matrices_torch, model_type="SparseGRF"):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.step_matrices_torch = step_matrices_torch
        
        # Select kernel based on model type
        if model_type == "SparseGRF":
            self.covar_module = SparseGRFKernel(
                max_walk_length=CONFIG["max_walk_length"],
                step_matrices_torch=self.step_matrices_torch
            )
        elif model_type == "SparseDiffusion":
            self.covar_module = SparseDiffusionKernel(
                max_walk_length=CONFIG["max_walk_length"],
                step_matrices_torch=self.step_matrices_torch
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.num_nodes = step_matrices_torch[0].shape[0]
        self.model_type = model_type

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x_test, n_samples=64):
        # Indices
        num_train = self.train_inputs[0].shape[0]
        train_indices = self.train_inputs[0].int().flatten()
        test_indices = x_test.int().flatten()

        # Features
        phi = self.covar_module._get_feature_matrix()  # [N, D]
        phi_train = phi[train_indices, :]              # [Ntr, D]
        phi_test = phi[test_indices, :]                # [Nte, D]

        # Kernel blocks
        K_train_train = phi_train @ phi_train.T        # [Ntr, Ntr]
        K_test_train = phi_test @ phi_train.T          # [Nte, Ntr]

        # Noise and system
        noise_variance = self.likelihood.noise.item()
        noise_std = float(np.sqrt(noise_variance))
        A = K_train_train + noise_variance * IdentityLinearOperator(num_train, device=x_test.device)

        # Pathwise prior samples (shared eps1 for train/test for correlation)
        eps1_batch = torch.randn(n_samples, self.num_nodes, device=x_test.device)
        eps2_batch = noise_std * torch.randn(n_samples, num_train, device=x_test.device)

        f_test_prior_batch = eps1_batch @ phi_test.T    # [B, Nte]
        f_train_prior_batch = eps1_batch @ phi_train.T  # [B, Ntr]

        # Solve for correction per batch rhs (CG handles multi-RHS via shape)
        b_batch = self.train_targets.unsqueeze(0) - (f_train_prior_batch + eps2_batch)  # [B, Ntr]
        v_batch = linear_cg(matmul_closure=A._matmul, rhs=b_batch.T, tolerance=settings.cg_tolerance.value())
        # v_batch: [Ntr, B]

        f_test_posterior_batch = f_test_prior_batch + (K_test_train @ v_batch).T  # [B, Nte]
        return f_test_posterior_batch

# ----------------------------
# Training & Metrics
# ----------------------------
def compute_rmse(y_true, y_pred):
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_nlpd(y_true, y_mean, y_std):
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_mean = y_mean.cpu().numpy() if isinstance(y_mean, torch.Tensor) else y_mean
    y_std = y_std.cpu().numpy() if isinstance(y_std, torch.Tensor) else y_std
    log_probs = -0.5 * np.log(2 * np.pi * y_std**2) - 0.5 * ((y_true - y_mean) / y_std)**2
    return float(-np.mean(log_probs))

def train_model(model, likelihood, data, lr=0.01, max_iter=100, seed=None, wpn=None, model_type=None):
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    train_losses = []

    # tqdm bar with manual updates
    pbar = tqdm(total=max_iter, desc=f"Training {model_type} Seed={seed}, WPN={wpn}", leave=False)

    for i in range(max_iter):
        optimizer.zero_grad()
        train_output = model(data['X_train'])
        train_loss = -mll(train_output, data['y_train'])
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        # update + detailed log every print_interval or at the end
        if (i + 1) % CONFIG["print_interval"] == 0 or (i + 1) == max_iter:
            pbar.update(CONFIG["print_interval"] if (i + 1) < max_iter else (max_iter % CONFIG["print_interval"]))
            mod = model.covar_module.modulator_vector.detach().cpu().numpy()
            log_msg = (
                f"[Seed {seed}, WPN {wpn}] "
                f"Iter {i+1:04d}/{max_iter} "
                f"Loss={train_loss:.3f}, Noise={likelihood.noise.item():.4f}, "
                f"Modulator={np.round(mod, 3)}"
            )
            tqdm.write(log_msg)
            pbar.set_postfix(loss=f"{train_loss.item():.3f}", noise=f"{likelihood.noise.item():.4f}")

    pbar.close()
    return train_losses


# ----------------------------
# Experiment loop
# ----------------------------
def run_experiment(seed, walks_per_node, model_type, data):
    tqdm.write(f"\n=== Running experiment: {model_type}, Seed={seed}, WPN={walks_per_node} ===")

    # Precompute step matrices
    pp = GraphPreprocessor(
        adjacency_matrix=data['A'],
        walks_per_node=walks_per_node,
        p_halt=CONFIG["p_halt"],
        max_walk_length=CONFIG["max_walk_length"],
        random_walk_seed=seed,
        load_from_disk=False,
        use_tqdm=True,
        n_processes=None
    )
    step_matrices_torch = pp.preprocess_graph(save_to_disk=False)
    step_matrices_torch = [step_matrix.to(device) for step_matrix in step_matrices_torch]

    # Model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GraphGPModel(data['X_train'], data['y_train'], likelihood, step_matrices_torch, model_type=model_type).to(device)

    # Train
    train_losses = train_model(
        model, likelihood, data,
        lr=CONFIG["learning_rate"],
        max_iter=CONFIG["num_iterations"],
        seed=seed, wpn=walks_per_node, model_type=model_type
    )

    # Inference
    model.eval(); likelihood.eval()
    with torch.no_grad():
        X_all = torch.arange(len(data['y']), dtype=torch.long, device=device).unsqueeze(1)
        all_samples = model.predict(X_all, n_samples=CONFIG["pathwise_samples"])
        all_mean = all_samples.mean(dim=0)
        all_std = all_samples.std(dim=0)

    # Metrics
    train_indices = data['X_train'].int().flatten().cpu().numpy()
    all_indices = np.arange(len(data['y']))
    test_indices = np.setdiff1d(all_indices, train_indices)

    test_rmse = compute_rmse(data['y'][test_indices], all_mean[test_indices])
    test_nlpd = compute_nlpd(data['y'][test_indices], all_mean[test_indices], all_std[test_indices])

    tqdm.write(
        f"[RESULT] {model_type}, Seed={seed}, WPN={walks_per_node} "
        f"→ Test RMSE={test_rmse:.4f}, Test NLPD={test_nlpd:.4f}"
    )

    # Cleanup
    del model, likelihood, step_matrices_torch, pp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        'seed': seed,
        'walks_per_node': walks_per_node,
        'model': model_type,
        'test_rmse': test_rmse,
        'test_nlpd': test_nlpd,
        'train_loss_final': train_losses[-1],
        'noise_variance': float(np.nan),
        'num_train': len(train_indices),
        'num_test': len(test_indices)
    }

# ----------------------------
# Main orchestration
# ----------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    total_experiments = len(CONFIG["seeds"]) * len(CONFIG["walks_per_nodes"]) * len(CONFIG["models"])
    print(f"Running {total_experiments} experiments...")

    for seed in tqdm(CONFIG["seeds"], desc="Seeds"):
        data = load_and_prepare_data(seed)

        for wpn in tqdm(CONFIG["walks_per_nodes"], desc=f"Seed {seed} - WPN", leave=False):
            for model_type in CONFIG["models"]:
                try:
                    result = run_experiment(seed, wpn, model_type, data)
                    results.append(result)
                except Exception as e:
                    tqdm.write(f"[ERROR] {model_type}, Seed={seed}, WPN={wpn}: {e}")
                    results.append({
                        'seed': seed, 'walks_per_node': wpn, 'model': model_type,
                        'test_rmse': np.nan, 'test_nlpd': np.nan,
                        'train_loss_final': np.nan, 'noise_variance': np.nan,
                        'num_train': np.nan, 'num_test': np.nan,
                        'error': str(e)
                    })

        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(results)
    
    # Create timestamped filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"wind_experiment_results_{timestamp}.csv")
    df.to_csv(results_file, index=False)

    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Results saved to: {results_file}")
    print(df.groupby(['model', 'walks_per_node'])[['test_rmse','test_nlpd']].mean().round(4))

if __name__ == "__main__":
    main()
