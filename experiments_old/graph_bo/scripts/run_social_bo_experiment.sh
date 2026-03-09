#!/bin/bash
# filepath: /scratches/cartwright/mz473/Efficient-Gaussian-Process-on-Graphs/graph_bo/scripts/run_social_bo_experiment.sh

# Social Network Bayesian Optimization Experiment Runner
# This script runs BO experiments using YAML configuration

# Default config path (can be overridden)
CONFIG_PATH="../configs/default_config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config path/to/config.yaml]"
            exit 1
            ;;
    esac
done

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to script directory
cd "$SCRIPT_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Configuration file not found: $CONFIG_PATH"
    echo "Using default configuration..."
    python run_graph_bo.py
else
    echo "Using configuration: $CONFIG_PATH"
    python run_graph_bo.py --config "$CONFIG_PATH"
fi