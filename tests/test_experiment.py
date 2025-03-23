import numpy as np
import pytest
from isd_lib.utils.experiment import Experiment
from isd_lib.algorithms.prange import PrangeISD
from isd_lib.algorithms.stern import SternISD

def test_experiment_init():
    """Test initializing an experiment."""
    # Initialize with no algorithms
    experiment = Experiment()
    assert len(experiment.algorithms) == 0
    
    # Initialize with algorithms
    prange = PrangeISD()
    stern = SternISD()
    experiment = Experiment([prange, stern])
    assert len(experiment.algorithms) == 2

def test_add_algorithm():
    """Test adding algorithms to an experiment."""
    experiment = Experiment()
    
    # Add one algorithm
    prange = PrangeISD()
    experiment.add_algorithm(prange)
    assert len(experiment.algorithms) == 1
    assert experiment.algorithms[0] == prange
    
    # Add another algorithm
    stern = SternISD()
    experiment.add_algorithm(stern)
    assert len(experiment.algorithms) == 2
    assert experiment.algorithms[1] == stern

def test_generate_random_instance():
    """Test generating a random syndrome decoding instance."""
    experiment = Experiment()
    
    # Try with different parameters if one fails
    for n, k in [(10, 5), (14, 7), (20, 10)]:
        try:
            t = 2
            H, s, e, t = experiment.generate_random_instance(n, k, t)
            
            # Check shapes
            assert H.shape == (n - k, n)
            assert s.shape == (n - k,)
            assert e.shape == (n,)
            
            # Check that e has weight t
            assert np.sum(e) == t
            
            # Check that s is the syndrome of e
            assert np.array_equal(np.dot(H, e) % 2, s)
            
            # If we got here, the test passed, so we can break
            break
        except ValueError:
            # If we get a ValueError, try with different parameters
            continue
    else:
        # If all parameter combinations fail, mark the test as failed
        pytest.fail("Could not generate a valid random instance with any parameter combination")

def test_run_trial():
    """Test running a single trial of an algorithm."""
    # Create a simple instance where decoding should succeed quickly
    n, k, t = 8, 4, 1
    
    prange = PrangeISD(max_iterations=1000)
    experiment = Experiment([prange])
    
    # Run a trial
    result = experiment.run_trial(n, k, t, 0)
    
    # Check that the result has all required keys
    assert 'success' in result
    assert 'iterations' in result
    assert 'time' in result
    assert 'weight' in result
    
    # If the trial was successful, check the weight
    if result['success']:
        assert result['weight'] <= t

def test_run_experiment():
    """Test running an experiment with multiple trials."""
    # Use small parameters for quick testing
    n, k, t = 8, 4, 1
    num_trials = 2
    
    prange = PrangeISD(max_iterations=100)
    stern = SternISD(max_iterations=10, p=1, l=0)
    
    experiment = Experiment([prange, stern])
    
    # Run the experiment
    results = experiment.run_experiment(n, k, t, num_trials)
    
    # Check that we have results for both algorithms
    assert 'PrangeISD' in results
    assert 'SternISD' in results
    
    # Check that we have the correct number of trials
    assert len(results['PrangeISD']) == num_trials
    assert len(results['SternISD']) == num_trials

def test_analyze_results():
    """Test analyzing experiment results."""
    # Create some mock results
    experiment = Experiment()
    experiment.results = {
        'PrangeISD': [
            {'success': True, 'iterations': 10, 'time': 0.1, 'weight': 1},
            {'success': True, 'iterations': 20, 'time': 0.2, 'weight': 1},
            {'success': False, 'iterations': 100, 'time': 1.0, 'weight': np.inf}
        ],
        'SternISD': [
            {'success': True, 'iterations': 5, 'time': 0.05, 'weight': 1},
            {'success': False, 'iterations': 10, 'time': 0.1, 'weight': np.inf},
            {'success': False, 'iterations': 10, 'time': 0.1, 'weight': np.inf}
        ]
    }
    
    # Analyze the results
    analysis = experiment.analyze_results()
    
    # Check that we have analysis for both algorithms
    assert 'PrangeISD' in analysis
    assert 'SternISD' in analysis
    
    # Check the Prange analysis
    prange_analysis = analysis['PrangeISD']
    assert prange_analysis['success_rate'] == 2/3
    assert prange_analysis['avg_time'] == pytest.approx(0.15)  # (0.1 + 0.2) / 2
    assert prange_analysis['avg_iterations'] == 15  # (10 + 20) / 2
    assert prange_analysis['trials'] == 3
    
    # Check the Stern analysis
    stern_analysis = analysis['SternISD']
    assert stern_analysis['success_rate'] == 1/3
    assert stern_analysis['avg_time'] == pytest.approx(0.05)
    assert stern_analysis['avg_iterations'] == 5 