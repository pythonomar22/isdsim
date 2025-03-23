#!/usr/bin/env python
"""
Example script to compare Prange's and Stern's ISD algorithms.

This script demonstrates how to use the ISD library to:
1. Create random linear codes
2. Run experiments with different ISD algorithms
3. Analyze and visualize the results

Usage:
    python isd_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Add the parent directory to the path to import the library
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isd_lib.core.linear_code import LinearCode
from isd_lib.algorithms.prange import PrangeISD
from isd_lib.algorithms.stern import SternISD
from isd_lib.utils.experiment import Experiment


def basic_example():
    """Basic example of using the ISD library."""
    # Create a linear code
    n, k = 20, 10
    code = LinearCode(n, k)
    
    # Create a random error vector of weight 2
    e = np.zeros(n, dtype=int)
    error_positions = np.random.choice(n, 2, replace=False)
    e[error_positions] = 1
    
    # Calculate the syndrome
    s = code.syndrome(e)
    
    print(f"Created a linear code with n={n}, k={k}")
    print(f"Generated an error vector with weight {np.sum(e)}")
    print(f"Error positions: {np.where(e == 1)[0]}")
    print(f"Syndrome: {s}")
    
    # Decode with Prange's algorithm
    prange = PrangeISD(max_iterations=1000)
    print("\nDecoding with Prange's algorithm...")
    start_time = time.time()
    e_prange, success_prange, iterations_prange = prange.decode(code.H, s, 2)
    time_prange = time.time() - start_time
    
    if success_prange:
        print(f"  Success! Found an error vector with weight {np.sum(e_prange)} in {iterations_prange} iterations.")
        print(f"  Error positions: {np.where(e_prange == 1)[0]}")
        print(f"  Time taken: {time_prange:.4f} seconds")
    else:
        print(f"  Failed to find a solution after {iterations_prange} iterations.")
    
    # Decode with Stern's algorithm
    stern = SternISD(max_iterations=100, p=1, l=0)
    print("\nDecoding with Stern's algorithm...")
    start_time = time.time()
    e_stern, success_stern, iterations_stern = stern.decode(code.H, s, 2)
    time_stern = time.time() - start_time
    
    if success_stern:
        print(f"  Success! Found an error vector with weight {np.sum(e_stern)} in {iterations_stern} iterations.")
        print(f"  Error positions: {np.where(e_stern == 1)[0]}")
        print(f"  Time taken: {time_stern:.4f} seconds")
    else:
        print(f"  Failed to find a solution after {iterations_stern} iterations.")


def run_experiment():
    """Run a systematic experiment to compare ISD algorithms."""
    # Set up the experiment
    n, k, t = 24, 12, 2
    num_trials = 10
    
    # Create the algorithms
    prange = PrangeISD(max_iterations=1000)
    stern1 = SternISD(max_iterations=100, p=1, l=0)
    stern2 = SternISD(max_iterations=100, p=2, l=4)
    
    # Create the experiment and add the algorithms
    experiment = Experiment([prange, stern1, stern2])
    
    # Run the experiment
    print(f"\nRunning experiment with n={n}, k={k}, t={t}, trials={num_trials}...")
    results = experiment.run_experiment(n, k, t, num_trials)
    
    # Analyze the results
    analysis = experiment.analyze_results()
    
    # Print the results
    print("\nExperiment results:")
    for algorithm_name, algorithm_analysis in analysis.items():
        print(f"\n{algorithm_name}:")
        print(f"  Success rate: {algorithm_analysis['success_rate']:.2f}")
        
        if algorithm_analysis['avg_time'] == np.inf:
            print("  Average time: N/A (no successful trials)")
        else:
            print(f"  Average time: {algorithm_analysis['avg_time']:.4f} seconds")
        
        if algorithm_analysis['avg_iterations'] == np.inf:
            print("  Average iterations: N/A (no successful trials)")
        else:
            print(f"  Average iterations: {algorithm_analysis['avg_iterations']:.1f}")
    
    # Visualize the results
    fig, axes = experiment.plot_summary()
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/experiment_results.png', dpi=300, bbox_inches='tight')
    
    print("\nResults saved to results/experiment_results.png")
    
    # Show the figure (if running in an interactive environment)
    plt.show()


if __name__ == "__main__":
    # Create the examples directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run the examples
    print("=== Basic Example ===")
    basic_example()
    
    print("\n=== Systematic Experiment ===")
    run_experiment() 