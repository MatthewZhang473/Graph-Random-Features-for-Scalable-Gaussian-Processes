import os
import json
from datetime import datetime

def save_results(results_df, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gt_best_value = float(results_df['ground_truth_best'].iloc[0]) if len(results_df) > 0 else None
    
    main_file = os.path.join(config.RESULTS_DIR, 'bo_experiment_results.csv')
    timestamped_file = os.path.join(config.RESULTS_DIR, f'bo_experiment_results_{timestamp}.csv')
    
    results_df.to_csv(main_file, index=False)
    results_df.to_csv(timestamped_file, index=False)
    
    config_summary = {
        'timestamp': timestamp,
        'total_experiments': len(results_df),
        'n_nodes': config.N_NODES,
        'num_bo_iterations': config.NUM_BO_ITERATIONS,
        'initial_points': config.INITIAL_POINTS,
        'batch_size': config.BATCH_SIZE,
        'num_bo_runs': config.NUM_BO_RUNS,
        'bo_seeds': config.BO_SEEDS,
        'data_seed': config.DATA_SEED,
        'algorithms': sorted(results_df['algorithm'].unique().tolist()),
        'ground_truth_best': gt_best_value
    }
    
    config_file = os.path.join(config.RESULTS_DIR, f'bo_experiment_config_{timestamp}.json')
    with open(config_file, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"📁 Results saved: {main_file}")
    return main_file

def print_summary(results_df, config):
    print("\n📊 Final Performance Summary:")
    print("=" * 50)
    
    final_iteration = results_df['iteration'].max()
    final_results = results_df[results_df['iteration'] == final_iteration]
    
    for algorithm in final_results['algorithm'].unique():
        algo_final = final_results[final_results['algorithm'] == algorithm]
        mean_best = algo_final['best_value'].mean()
        std_best = algo_final['best_value'].std()
        mean_regret = algo_final['regret'].mean()
        std_regret = algo_final['regret'].std()
        
        print(f"\n{algorithm.upper()}:")
        print(f"  Best Value: {mean_best:.4f} ± {std_best:.4f}")
        print(f"  Final Regret: {mean_regret:.4f} ± {std_regret:.4f}")
        print(f"  Success Rate: {(algo_final['regret'] < 0.1).mean()*100:.1f}% (regret < 0.1)")
