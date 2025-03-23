import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from ..core.linear_code import LinearCode

class Experiment:
    """
    A class for running systematic experiments to compare the performance of different ISD algorithms.
    
    This class provides methods for:
    - Generating random linear codes
    - Creating random error vectors
    - Running multiple trials of different ISD algorithms
    - Collecting and analyzing results
    - Visualizing performance comparisons
    """
    
    def __init__(self, algorithms=None):
        """
        Initialize an experiment.
        
        Args:
            algorithms (list, optional): A list of ISD algorithm instances to compare.
        """
        self.algorithms = algorithms or []
        self.results = defaultdict(list)
    
    def add_algorithm(self, algorithm):
        """
        Add an ISD algorithm to the experiment.
        
        Args:
            algorithm: An instance of a class inheriting from BaseISD
        """
        self.algorithms.append(algorithm)
    
    def generate_random_instance(self, n, k, t):
        """
        Generate a random instance of the syndrome decoding problem.
        
        Args:
            n (int): Code length
            k (int): Code dimension
            t (int): Error weight
            
        Returns:
            tuple: (H, s, e, t)
                - H (numpy.ndarray): Parity-check matrix
                - s (numpy.ndarray): Syndrome
                - e (numpy.ndarray): Error vector (for verification)
                - t (int): Error weight
        """
        # Generate a random linear code
        code = LinearCode(n, k)
        
        # Generate a random error vector of weight t
        e = np.zeros(n, dtype=int)
        error_positions = np.random.choice(n, t, replace=False)
        e[error_positions] = 1
        
        # Calculate the syndrome
        s = code.syndrome(e)
        
        return code.H, s, e, t
    
    def run_trial(self, n, k, t, algorithm_idx, max_time=30):
        """
        Run a single trial of an algorithm on a random instance.
        
        Args:
            n (int): Code length
            k (int): Code dimension
            t (int): Error weight
            algorithm_idx (int): Index of the algorithm to use
            max_time (float, optional): Maximum time (in seconds) to allow for decoding
            
        Returns:
            dict: Trial results with keys:
                - 'success': Whether decoding was successful
                - 'iterations': Number of iterations performed
                - 'time': Time taken in seconds
                - 'weight': Weight of the found error vector (if successful)
        """
        # Generate a random instance
        H, s, e_true, t = self.generate_random_instance(n, k, t)
        
        # Get the algorithm
        algorithm = self.algorithms[algorithm_idx]
        
        # Reset statistics
        algorithm.reset_statistics()
        
        # Run the algorithm with a time limit
        start_time = time.time()
        
        try:
            e, success, iterations = algorithm.decode(H, s, t)
            time_taken = time.time() - start_time
            
            if time_taken > max_time:
                # Time limit exceeded
                return {
                    'success': False,
                    'iterations': iterations,
                    'time': time_taken,
                    'weight': np.inf
                }
            
            # Check the solution
            if success and not algorithm.check_solution(H, s, e, t):
                success = False
            
            # Calculate the weight if successful
            weight = np.sum(e) if success else np.inf
            
            return {
                'success': success,
                'iterations': iterations,
                'time': time_taken,
                'weight': weight
            }
            
        except Exception as ex:
            # Handle any exceptions during decoding
            return {
                'success': False,
                'iterations': 0,
                'time': time.time() - start_time,
                'weight': np.inf,
                'error': str(ex)
            }
    
    def run_experiment(self, n, k, t, num_trials=10, max_time=30):
        """
        Run an experiment with multiple trials comparing all algorithms.
        
        Args:
            n (int): Code length
            k (int): Code dimension
            t (int): Error weight
            num_trials (int, optional): Number of trials to run
            max_time (float, optional): Maximum time (in seconds) to allow for each trial
            
        Returns:
            dict: A dictionary with algorithm names as keys and lists of trial results as values
        """
        results = defaultdict(list)
        
        for trial in range(num_trials):
            # Generate a random instance
            H, s, e_true, t = self.generate_random_instance(n, k, t)
            
            for i, algorithm in enumerate(self.algorithms):
                # Reset statistics
                algorithm.reset_statistics()
                
                # Run the algorithm with a time limit
                start_time = time.time()
                
                try:
                    e, success, iterations = algorithm.decode(H, s, t)
                    time_taken = time.time() - start_time
                    
                    if time_taken > max_time:
                        # Time limit exceeded
                        result = {
                            'success': False,
                            'iterations': iterations,
                            'time': time_taken,
                            'weight': np.inf
                        }
                    else:
                        # Check the solution
                        if success and not algorithm.check_solution(H, s, e, t):
                            success = False
                        
                        # Calculate the weight if successful
                        weight = np.sum(e) if success else np.inf
                        
                        result = {
                            'success': success,
                            'iterations': iterations,
                            'time': time_taken,
                            'weight': weight
                        }
                
                except Exception as ex:
                    # Handle any exceptions during decoding
                    result = {
                        'success': False,
                        'iterations': 0,
                        'time': time.time() - start_time,
                        'weight': np.inf,
                        'error': str(ex)
                    }
                
                # Add the result to our collection
                results[algorithm.__class__.__name__].append(result)
                
        # Store the results
        self.results = results
        
        return results
    
    def analyze_results(self):
        """
        Analyze the results of the experiment.
        
        Returns:
            dict: A dictionary with algorithm names as keys and analysis as values
        """
        analysis = {}
        
        for algorithm_name, trials in self.results.items():
            # Calculate success rate
            success_rate = sum(1 for t in trials if t['success']) / len(trials) if trials else 0
            
            # Calculate average time (for successful trials)
            successful_trials = [t for t in trials if t['success']]
            avg_time = sum(t['time'] for t in successful_trials) / len(successful_trials) if successful_trials else np.inf
            
            # Calculate average iterations (for successful trials)
            avg_iterations = sum(t['iterations'] for t in successful_trials) / len(successful_trials) if successful_trials else np.inf
            
            analysis[algorithm_name] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'avg_iterations': avg_iterations,
                'trials': len(trials)
            }
        
        return analysis
    
    def plot_success_rate(self, ax=None, **kwargs):
        """
        Plot the success rate of each algorithm.
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on.
            **kwargs: Additional arguments to pass to plot.
            
        Returns:
            matplotlib.axes.Axes: The axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        analysis = self.analyze_results()
        
        names = list(analysis.keys())
        success_rates = [analysis[name]['success_rate'] for name in names]
        
        ax.bar(names, success_rates, **kwargs)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Success Rate')
        ax.set_title('Algorithm Success Rate Comparison')
        ax.set_ylim(0, 1.1)
        
        for i, rate in enumerate(success_rates):
            ax.text(i, rate + 0.05, f'{rate:.2f}', ha='center')
        
        return ax
    
    def plot_avg_time(self, ax=None, **kwargs):
        """
        Plot the average time of each algorithm (for successful trials).
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on.
            **kwargs: Additional arguments to pass to plot.
            
        Returns:
            matplotlib.axes.Axes: The axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        analysis = self.analyze_results()
        
        names = list(analysis.keys())
        times = [analysis[name]['avg_time'] for name in names]
        
        # Filter out infinite times
        valid_indices = [i for i, t in enumerate(times) if t != np.inf]
        valid_names = [names[i] for i in valid_indices]
        valid_times = [times[i] for i in valid_indices]
        
        ax.bar(valid_names, valid_times, **kwargs)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Average Time (s)')
        ax.set_title('Algorithm Average Time Comparison (Successful Trials)')
        
        for i, time in enumerate(valid_times):
            ax.text(i, time + 0.05, f'{time:.4f}s', ha='center')
        
        return ax
    
    def plot_avg_iterations(self, ax=None, **kwargs):
        """
        Plot the average iterations of each algorithm (for successful trials).
        
        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on.
            **kwargs: Additional arguments to pass to plot.
            
        Returns:
            matplotlib.axes.Axes: The axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        analysis = self.analyze_results()
        
        names = list(analysis.keys())
        iterations = [analysis[name]['avg_iterations'] for name in names]
        
        # Filter out infinite iterations
        valid_indices = [i for i, it in enumerate(iterations) if it != np.inf]
        valid_names = [names[i] for i in valid_indices]
        valid_iterations = [iterations[i] for i in valid_indices]
        
        ax.bar(valid_names, valid_iterations, **kwargs)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Average Iterations')
        ax.set_title('Algorithm Average Iterations Comparison (Successful Trials)')
        
        for i, iters in enumerate(valid_iterations):
            ax.text(i, iters + 0.05, f'{iters:.1f}', ha='center')
        
        return ax
    
    def plot_summary(self, figsize=(15, 10)):
        """
        Create a summary plot with success rate, average time, and average iterations.
        
        Args:
            figsize (tuple, optional): Figure size
            
        Returns:
            tuple: (fig, axes) - The figure and axes objects
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        self.plot_success_rate(ax=axes[0], color='skyblue')
        self.plot_avg_time(ax=axes[1], color='lightgreen')
        self.plot_avg_iterations(ax=axes[2], color='salmon')
        
        plt.tight_layout()
        
        return fig, axes 