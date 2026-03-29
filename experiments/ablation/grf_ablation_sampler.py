"""Ablation random-walk sampler that removes the `degree / (1-p_halt)` factor from the load update."""

import os
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import numpy as np
from tqdm.auto import tqdm

from grf_gp.sampler import GRFSampler, _init_worker
from grf_gp.utils.csr import build_csr_from_entries
from grf_gp.utils.sparse_lo import SparseLinearOperator
import grf_gp.sampler as sampler_mod


def _run_walks_ablation(
    nodes: np.ndarray,
    walks_per_node: int,
    p_halt: float,
    max_walk_length: int,
    seed: int,
    show_progress: bool,
) -> List[defaultdict]:
    """Run the ablated walk recursion without the ``1 - p_halt`` factor."""
    if (
        sampler_mod._G_CROW is None
        or sampler_mod._G_COL is None
        or sampler_mod._G_DATA is None
    ):
        raise RuntimeError("CSR arrays are not available in this process.")

    crow = sampler_mod._G_CROW
    col = sampler_mod._G_COL
    data = sampler_mod._G_DATA

    step_accumulators = [defaultdict(float) for _ in range(max_walk_length)]
    iterator = tqdm(nodes, desc="Process walks", disable=not show_progress)

    for start_node in iterator:
        start_node = int(start_node)
        rng = np.random.default_rng(seed + start_node)
        for _ in range(walks_per_node):
            current_node = start_node
            load = 1.0
            for step in range(max_walk_length):
                step_accumulators[step][(start_node, current_node)] += load

                start = crow[current_node]
                end = crow[current_node + 1]
                degree = end - start
                if degree == 0:
                    break
                if rng.random() < p_halt:
                    break

                offset = rng.integers(degree)
                weight = data[start + offset]
                current_node = int(col[start + offset])
                load *= weight

    return step_accumulators


def _worker_walks_ablation(args: tuple) -> List[defaultdict]:
    """Worker wrapper for the ablation random-walk loop."""
    (
        nodes,
        walks_per_node,
        p_halt,
        max_walk_length,
        seed,
        show_progress,
    ) = args
    return _run_walks_ablation(
        nodes=np.asarray(nodes),
        walks_per_node=walks_per_node,
        p_halt=p_halt,
        max_walk_length=max_walk_length,
        seed=seed,
        show_progress=show_progress,
    )


class GRFAblationSampler(GRFSampler):
    """GRF sampler using the ablated load update from the old study."""

    def sample_random_walk_matrices(self) -> List[SparseLinearOperator]:
        crow_indices = self.adjacency_csr.crow_indices().numpy()
        col_indices = self.adjacency_csr.col_indices().numpy()
        values = self.adjacency_csr.values().numpy()
        num_nodes = self.adjacency_csr.size(0)

        n_proc = self.n_processes or os.cpu_count() or 1
        chunks = np.array_split(np.arange(num_nodes), n_proc)
        ctx = mp.get_context("fork")

        with ProcessPoolExecutor(
            max_workers=n_proc,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(crow_indices, col_indices, values),
        ) as executor:
            args = [
                (
                    chunk.tolist(),
                    self.walks_per_node,
                    self.p_halt,
                    self.max_walk_length,
                    self.seed + i,
                    self.use_tqdm and i == 0,
                )
                for i, chunk in enumerate(chunks)
            ]
            futures = [executor.submit(_worker_walks_ablation, a) for a in args]
            results = [fut.result() for fut in as_completed(futures)]

        accumulators = [defaultdict(float) for _ in range(self.max_walk_length)]
        for result in results:
            for step in range(self.max_walk_length):
                for key, val in result[step].items():
                    accumulators[step][key] += val

        return [
            SparseLinearOperator(
                build_csr_from_entries(num_nodes, acc) * (1.0 / self.walks_per_node)
            )
            for acc in accumulators
        ]
