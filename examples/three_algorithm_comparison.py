#!/usr/bin/env python
"""
Comparison of three ISD algorithms: Prange, Stern, and BKW.

This script demonstrates the performance differences between:
1. Prange's algorithm - The simplest ISD approach
2. Stern's algorithm - A meet-in-the-middle approach
3. BKW algorithm - A collision-finding approach

Usage:
    python three_algorithm_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import pandas as pd

# Add the parent directory to the path to import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isd_lib.core.linear_code import LinearCode
from isd_lib.algorithms.prange import PrangeISD
from isd_lib.algorithms.stern import SternISD
from isd_lib.algorithms.bkw import BKWISD
from isd_lib.utils.experiment import Experiment


def run_algorithm_comparison():
    """Run a comparison between all three algorithms."""
    print("Starting algorithm comparison...")
    
    # Create algorithms with comparable parameters
    prange = PrangeISD(max_iterations=1000)
    stern = SternISD(max_iterations=100, p=2, l=2)
    bkw = BKWISD(max_iterations=100, num_blocks=2)
    
    # Create experiment with all algorithms
    experiment = Experiment([prange, stern, bkw])
    
    # Run experiment with moderate-sized code
    n, k, t = 24, 12, 2
    num_trials = 10
    
    print(f"Running experiment with code parameters: n={n}, k={k}, t={t}")
    print(f"Number of trials: {num_trials}")
    
    results = experiment.run_experiment(n, k, t, num_trials)
    
    # Analyze results
    analysis = experiment.analyze_results()
    
    # Create a DataFrame for easier viewing
    df = pd.DataFrame(analysis).T  # Transpose to make algorithm names the index
    
    print("\nResults summary:")
    print(df)
    
    # Plot results
    fig, axes = experiment.plot_summary(figsize=(15, 5))
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'three_algorithm_comparison.png'), dpi=300)
    
    print(f"\nResults figure saved to: {os.path.join(output_dir, 'three_algorithm_comparison.png')}")
    
    return fig, axes, df


def run_error_weight_comparison():
    """Compare algorithm performance across different error weights."""
    print("\nComparing performance across different error weights...")
    
    # Fixed code parameters
    n, k = 20, 10
    error_weights = [1, 2, 3]
    num_trials = 5
    
    # Prepare algorithms
    prange = PrangeISD(max_iterations=1000)
    stern = SternISD(max_iterations=100, p=2, l=2)
    bkw = BKWISD(max_iterations=100, num_blocks=2)
    
    # Store results
    all_results = []
    
    for t in error_weights:
        # Create a new experiment for each error weight
        experiment = Experiment([prange, stern, bkw])
        
        print(f"Running experiment with error weight t={t}")
        experiment.run_experiment(n, k, t, num_trials)
        
        # Analyze results
        analysis = experiment.analyze_results()
        
        # Store results with correct dictionary access
        for alg_name in analysis:
            row = {
                'Algorithm': alg_name,
                'Error Weight': t,
                'Success Rate': analysis[alg_name]['success_rate'],
                'Avg. Time (s)': analysis[alg_name]['avg_time'],
                'Avg. Iterations': analysis[alg_name]['avg_iterations']
            }
            all_results.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    print("\nResults by error weight:")
    print(results_df)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Group by algorithm
    for alg in ['Prange', 'Stern', 'BKW']:
        df_alg = results_df[results_df['Algorithm'] == alg]
        
        # Plot success rate
        axes[0].plot(df_alg['Error Weight'], df_alg['Success Rate'], 'o-', label=alg)
        
        # Plot avg time
        axes[1].plot(df_alg['Error Weight'], df_alg['Avg. Time (s)'], 'o-', label=alg)
        
        # Plot avg iterations
        axes[2].plot(df_alg['Error Weight'], df_alg['Avg. Iterations'], 'o-', label=alg)
    
    # Set labels and titles
    axes[0].set_xlabel('Error Weight (t)')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate vs. Error Weight')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel('Error Weight (t)')
    axes[1].set_ylabel('Average Time (s)')
    axes[1].set_title('Average Time vs. Error Weight')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].set_xlabel('Error Weight (t)')
    axes[2].set_ylabel('Average Iterations')
    axes[2].set_title('Average Iterations vs. Error Weight')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'error_weight_comparison.png'), dpi=300)
    
    print(f"Error weight comparison figure saved to: {os.path.join(output_dir, 'error_weight_comparison.png')}")
    
    return fig, axes, results_df


if __name__ == "__main__":
    # Run the standard comparison
    fig1, axes1, df1 = run_algorithm_comparison()
    
    # Run the error weight comparison
    fig2, axes2, df2 = run_error_weight_comparison()
    
    # Show the plots
    plt.show() 