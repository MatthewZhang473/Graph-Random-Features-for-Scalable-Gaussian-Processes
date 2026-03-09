import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from netCDF4 import Dataset
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, wgs84, utc


def load_wind_data(
    nc_file,
    use_downsampling=True,
    downsample_factor=10,
    dtype=np.float64,
):
    with Dataset(nc_file, mode="r") as dataset:
        lat = dataset.variables["latitude"][:].astype(dtype, copy=False)
        lon = dataset.variables["longitude"][:].astype(dtype, copy=False)
        u = dataset.variables["u"][:].astype(dtype, copy=False)
        v = dataset.variables["v"][:].astype(dtype, copy=False)

    u_500 = u[0, 0, :, :]
    v_500 = v[0, 0, :, :]

    if use_downsampling:
        lat = lat[::downsample_factor]
        lon = lon[::downsample_factor]
        u_500 = u_500[::downsample_factor, ::downsample_factor]
        v_500 = v_500[::downsample_factor, ::downsample_factor]

    return lat, lon, u_500, v_500


def deg2rad(x):
    return np.deg2rad(x)


def great_circle_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg, R=1.0):
    lat1, lon1 = deg2rad(lat1_deg), deg2rad(lon1_deg)
    lat2, lon2 = deg2rad(lat2_deg), deg2rad(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def build_sphere_grid_graph(lat, lon):
    n_lat = len(lat)
    n_lon = len(lon)
    rows, cols, data = [], [], []
    nbrs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for i in range(n_lat):
        for j in range(n_lon):
            nid = i * n_lon + j
            for di, dj in nbrs:
                ii = i + di
                jj = (j + dj) % n_lon
                if 0 <= ii < n_lat:
                    nid2 = ii * n_lon + jj
                    w = great_circle_distance(lat[i], lon[j], lat[ii], lon[jj], R=1.0)
                    rows.append(nid)
                    cols.append(nid2)
                    data.append(w)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_lat * n_lon, n_lat * n_lon))
    A = ((A + A.T) * 0.5).tocsr().astype(lat.dtype, copy=False)
    G = nx.from_scipy_sparse_array(A)
    return G, A


def generate_aeolus_track():
    line1 = "1 43600U 18066A   21153.73585495  .00031128  00000-0  12124-3 0  9990"
    line2 = "2 43600  96.7150 160.8035 0006915  90.4181 269.7884 15.87015039160910"
    ts = load.timescale()
    aeolus = EarthSatellite(line1, line2, "AEOLUS", ts)
    start = datetime(2019, 1, 1, 9, tzinfo=utc)
    stop = start + timedelta(hours=24)
    step = timedelta(minutes=1)
    times = []
    t = start
    while t <= stop:
        times.append(t)
        t += step
    geocentric = aeolus.at(ts.from_datetimes(times))
    lat, lon = wgs84.latlon_of(geocentric)
    return pd.DataFrame({"time": times, "lat": lat.degrees, "lon": lon.degrees % 360.0})


def nearest_node_indices_for_track(track_lat, track_lon, lat, lon):
    track_lat = np.asarray(track_lat)
    track_lon = np.asarray(track_lon) % 360.0
    i_idx = np.abs(track_lat[:, None] - lat[None, :]).argmin(axis=1)
    j_idx = np.abs(track_lon[:, None] - lon[None, :]).argmin(axis=1)
    return i_idx * len(lon) + j_idx


def prepare_wind_graph_data(
    nc_file,
    use_downsampling=True,
    downsample_factor=10,
    dtype=np.float64,
):
    lat, lon, u_500, v_500 = load_wind_data(
        nc_file,
        use_downsampling=use_downsampling,
        downsample_factor=downsample_factor,
        dtype=dtype,
    )
    G, A = build_sphere_grid_graph(lat, lon)
    track = generate_aeolus_track()
    train_idx = np.unique(
        nearest_node_indices_for_track(
            track["lat"].values,
            track["lon"].values,
            lat,
            lon,
        )
    )
    n_lat, n_lon = len(lat), len(lon)
    X = np.arange(n_lat * n_lon)
    y = np.sqrt((u_500 ** 2 + v_500 ** 2)).reshape(-1).astype(dtype, copy=False)
    y_mean = y[train_idx].mean()
    y_std = y[train_idx].std()
    y_norm = ((y - y_mean) / y_std).astype(dtype, copy=False)
    test_idx = np.setdiff1d(X, train_idx)
    X_train = X[train_idx]
    y_train = y_norm[train_idx]
    X_test = X[test_idx]
    y_test = y_norm[test_idx]

    return {
        "lat": lat,
        "lon": lon,
        "u_500": u_500,
        "v_500": v_500,
        "graph": G,
        "adjacency": A,
        "X": X,
        "y": y_norm,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_mean": y_mean,
        "y_std": y_std,
    }
