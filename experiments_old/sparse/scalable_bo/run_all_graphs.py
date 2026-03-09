#!/usr/bin/env python3
"""
Convenience script to run BO experiments on all graph types.
"""

import subprocess
import sys
import time

def run_experiment(graph_type, n_nodes=10000, n_runs=3, n_iterations=30):
    """Run experiment for a specific graph type."""
    print(f"\n{'='*60}")
    print(f"RUNNING {graph_type.upper()} GRAPH EXPERIMENT")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "experiments_sparse/scalable_bo/run_bo.py",
        "--graph-type", graph_type,
        "--n-nodes", str(n_nodes),
        "--n-runs", str(n_runs),
        "--n-iterations", str(n_iterations)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✅ {graph_type} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {graph_type} failed after {elapsed:.1f}s: {e}")
        return False

def main():
    # Configuration for all experiments - UPDATED
    n_nodes_per_type = {
        'circle': int(1e6),           # Can handle larger sizes efficiently
        'grid': int(1e6),            # 100x100 grid (perfect square)
        'periodic_grid': int(1e6),   # 100x100 periodic grid (torus)
        'staircase_grid': int(1e6),  # 100x100 grid with staircase function
        'grid_multimodal': int(1e6)  # 100x100 grid with multiple peaks
    }
    
    n_runs = 5
    n_iterations = 50
    
    print("🚀 Running BO experiments on all graph types")
    print(f"   Runs per type: {n_runs}")
    print(f"   BO iterations: {n_iterations}")
    print(f"   Node counts: {n_nodes_per_type}")
    
    results = {}
    total_start = time.time()
    
    for graph_type, n_nodes in n_nodes_per_type.items():
        success = run_experiment(graph_type, n_nodes, n_runs, n_iterations)
        results[graph_type] = success
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")
    
    for graph_type, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {graph_type}: {'SUCCESS' if success else 'FAILED'}")
    
    successful = sum(results.values())
    print(f"\nOverall: {successful}/{len(results)} experiments completed successfully")

if __name__ == "__main__":
    main()
