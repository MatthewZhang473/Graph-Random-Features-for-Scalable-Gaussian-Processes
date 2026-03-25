from pathlib import Path
import csv
import gzip

import networkx as nx
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "social_networks"

DATASETS = {
    "enron": DATA_DIR / "email-Enron.txt.gz",
    "youtube": DATA_DIR / "com-youtube.ungraph.txt.gz",
    "twitch": DATA_DIR / "large_twitch_edges.csv",
    "facebook": DATA_DIR / "musae_facebook_edges.csv",
}


def prepare_social_network_data(dataset_name: str, dtype=np.float64):
    path = DATASETS[dataset_name]
    graph = nx.Graph()

    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                u, v = map(int, line.split())
                graph.add_edge(u, v)
    else:
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                u, v = map(int, row[:2])
                graph.add_edge(u, v)

    nodelist = sorted(graph.nodes())
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodelist, format="csr")
    y = np.array([graph.degree(node) for node in nodelist], dtype=dtype)
    x = np.arange(len(nodelist), dtype=np.int64)

    return {
        "dataset_name": dataset_name,
        "graph": graph,
        "X": x,
        "y": y,
        "adjacency": adjacency,
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
    }
