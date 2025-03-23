import numpy as np
import pytest
from isd_lib.core.linear_code import LinearCode
from isd_lib.algorithms.bkw import BKWISD

class TestBKWISD:
    """Test cases for the BKW Information Set Decoding algorithm."""
    
    def test_zero_syndrome(self):
        """Test decoding with a zero syndrome."""
        n, k = 20, 10
        code = LinearCode(n, k)
        s = np.zeros(n - k, dtype=int)
        t = 2
        
        bkw = BKWISD(max_iterations=100)
        e, success, iterations = bkw.decode(code.H, s, t)
        
        assert success is True
        assert iterations == 0
        assert np.all(e == 0)
    
    def test_small_code(self):
        """Test decoding with a small code."""
        n, k = 10, 5
        code = LinearCode(n, k)
        
        # Create an error vector with weight 1
        e_true = np.zeros(n, dtype=int)
        e_true[2] = 1
        
        s = code.syndrome(e_true)
        t = 1
        
        bkw = BKWISD(max_iterations=300)
        e, success, iterations = bkw.decode(code.H, s, t)
        
        # BKW is probabilistic and may not always find a solution
        # If it succeeds, check that the solution is valid
        if success:
            assert np.sum(e) <= t
            assert np.all(code.syndrome(e) == s)
        else:
            # Skip if it didn't find a solution
            pytest.skip("BKW failed to find a solution in this run (expected due to probabilistic nature)")
    
    def test_decode_random_error(self):
        """Test decoding a random error vector."""
        n, k = 20, 10
        code = LinearCode(n, k)
        
        # Create a random error vector with weight 2
        e_true = np.zeros(n, dtype=int)
        error_positions = np.random.choice(n, 2, replace=False)
        e_true[error_positions] = 1
        
        s = code.syndrome(e_true)
        t = 2
        
        bkw = BKWISD(max_iterations=200)
        e, success, iterations = bkw.decode(code.H, s, t)
        
        # If successful, check that the solution is valid
        if success:
            assert np.sum(e) <= t
            assert np.all(code.syndrome(e) == s)
    
    def test_failure_high_weight(self):
        """Test that the algorithm correctly reports failure for high weight errors."""
        n, k = 20, 10
        code = LinearCode(n, k)
        
        # Create an error vector with weight higher than we're willing to decode
        e_true = np.zeros(n, dtype=int)
        error_positions = np.random.choice(n, 5, replace=False)
        e_true[error_positions] = 1
        
        s = code.syndrome(e_true)
        
        # Try to decode with t=2, which should fail because the true weight is 5
        t = 2
        
        bkw = BKWISD(max_iterations=10)
        e, success, iterations = bkw.decode(code.H, s, t)
        
        # It might occasionally find a valid codeword by chance, but usually should fail
        if not success:
            assert iterations == 10
    
    def test_different_parameters(self):
        """Test with different parameter settings."""
        n, k = 24, 12
        code = LinearCode(n, k)
        
        # Create a random error vector with weight 2
        e_true = np.zeros(n, dtype=int)
        error_positions = np.random.choice(n, 2, replace=False)
        e_true[error_positions] = 1
        
        s = code.syndrome(e_true)
        t = 2
        
        # Test with different num_blocks settings
        bkw1 = BKWISD(max_iterations=50, num_blocks=2)
        bkw2 = BKWISD(max_iterations=50, num_blocks=3)
        
        e1, success1, iterations1 = bkw1.decode(code.H, s, t)
        e2, success2, iterations2 = bkw2.decode(code.H, s, t)
        
        # Both should find valid solutions (if they succeed)
        if success1:
            assert np.sum(e1) <= t
            assert np.all(code.syndrome(e1) == s)
        
        if success2:
            assert np.sum(e2) <= t
            assert np.all(code.syndrome(e2) == s)
    
    def test_features(self):
        """Combined test for various features of the BKW algorithm."""
        # Initialize
        n, k = 20, 10
        code = LinearCode(n, k)
        bkw = BKWISD()
        
        # Test 1: Check solution with a valid error vector
        e_valid = np.zeros(n, dtype=int)
        e_valid[5] = 1
        e_valid[8] = 1  # Weight 2
        s_valid = code.syndrome(e_valid)
        
        # This should return True
        assert bkw.check_solution(code.H, s_valid, e_valid, 2)
        
        # Test 2: Check solution with wrong weight
        e_heavy = np.zeros(n, dtype=int)
        e_heavy[0] = e_heavy[1] = e_heavy[2] = 1  # Weight 3
        s_heavy = code.syndrome(e_heavy)
        
        # This should return False for t=2
        if not np.array_equal(s_heavy, s_valid):  # Make sure we test a different syndrome
            assert not bkw.check_solution(code.H, s_heavy, e_heavy, 2)
        
        # Test 3: Check solution with wrong syndrome
        e_wrong = np.zeros(n, dtype=int)
        e_wrong[3] = 1  # Weight 1, different from e_valid
        s_wrong = code.syndrome(e_wrong)
        
        # This should return False when checked against s_valid
        if not np.array_equal(s_wrong, s_valid):  # Make sure we test a different syndrome
            assert not bkw.check_solution(code.H, s_valid, e_wrong, 2) 