import argparse

def parse_bo_args():
    """Parse command line arguments for social network BO experiments"""
    
    parser = argparse.ArgumentParser(description='Run Bayesian Optimization on social networks')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['facebook', 'youtube', 'twitch', 'enron'], 
                       default=['facebook'],
                       help='Datasets to run experiments on')
    parser.add_argument('--algorithms', nargs='+',
                       choices=['random_search', 'bfs', 'sparse_grf'],
                       default=['random_search', 'bfs', 'sparse_grf'],
                       help='BO algorithms to compare')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of BO iterations')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of BO runs per algorithm')
    parser.add_argument('--initial-points', type=int, default=100,
                       help='Number of initial random points')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for BO')
    
    return parser.parse_args()
